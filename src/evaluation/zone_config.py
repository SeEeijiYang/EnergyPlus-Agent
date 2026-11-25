"""Locate zone configuration and identify the static evidence for candidate routes."""

from dataclasses import asdict, dataclass, field
from typing import Any

from idfpy import IDFBaseModel
from idfpy.models import (
    AirflowNetworkMultiZoneSurface,
    AirflowNetworkMultiZoneZone,
    AirflowNetworkSimulationControl,
    AirLoopHVAC,
    BuildingSurfaceDetailed,
    ControllerMechanicalVentilation,
    ControllerOutdoorAir,
    FenestrationSurfaceDetailed,
    HVACTemplateZoneIdealLoadsAirSystem,
    People,
    SizingZone,
    Space,
    SpaceList,
    Zone,
    ZoneControlThermostat,
    ZoneHVACAirDistributionUnit,
    ZoneHVACEquipmentConnections,
    ZoneHVACEquipmentList,
    ZoneList,
    ZoneVentilationWindandStackOpenArea,
)

from src.evaluation.idf_input import Finding, ParsedIDF
from src.evaluation.reference_checks import NameChecks

_TERMINALS = {
    "AirTerminal:SingleDuct:VAV:Reheat",
    "AirTerminal:SingleDuct:VAV:NoReheat",
    "AirTerminal:SingleDuct:SeriesPIU:Reheat",
    "AirTerminal:SingleDuct:ParallelPIU:Reheat",
}
_IDEAL_LOADS = "ZoneHVAC:IdealLoadsAirSystem"
_SUPPORTED_ZONE_HVAC = {
    _IDEAL_LOADS,
    "ZoneHVAC:AirDistributionUnit",
    "ZoneHVAC:EquipmentConnections",
    "ZoneHVAC:EquipmentList",
    "HVACTemplate:Zone:IdealLoadsAirSystem",
}
_OPENING_TYPES = {
    "AirflowNetwork:MultiZone:Component:DetailedOpening",
    "AirflowNetwork:MultiZone:Component:SimpleOpening",
    "AirflowNetwork:MultiZone:Component:HorizontalOpening",
}
_PEOPLE_FIELDS = (
    "zone_or_zonelist_or_space_or_spacelist_name",
    "number_of_people_calculation_method",
    "number_of_people",
    "people_per_floor_area",
    "floor_area_per_person",
    "number_of_people_schedule_name",
    "activity_level_schedule_name",
    "work_efficiency_schedule_name",
    "clothing_insulation_calculation_method",
    "clothing_insulation_calculation_method_schedule_name",
    "clothing_insulation_schedule_name",
    "air_velocity_schedule_name",
    "mean_radiant_temperature_calculation_type",
    *(f"thermal_comfort_model_{number}_type" for number in range(1, 8)),
)
_OUTDOOR_AIR_FIELDS = (
    "outdoor_air_method",
    "outdoor_air_flow_per_person",
    "outdoor_air_flow_per_zone_floor_area",
    "outdoor_air_flow_per_zone",
    "outdoor_air_flow_air_changes_per_hour",
    "outdoor_air_schedule_name",
)


@dataclass
class ZoneConfiguration:
    zone: dict[str, Any]
    spaces: list[dict[str, Any]] = field(default_factory=list)
    people: list[dict[str, Any]] = field(default_factory=list)
    thermostats: list[dict[str, Any]] = field(default_factory=list)
    equipment: list[dict[str, Any]] = field(default_factory=list)
    outdoor_air_design: list[dict[str, Any]] = field(default_factory=list)
    mechanical_ventilation: list[dict[str, Any]] = field(default_factory=list)
    exterior_fenestration: list[dict[str, Any]] = field(default_factory=list)
    opening_controls: list[dict[str, Any]] = field(default_factory=list)

    def report(self) -> dict[str, Any]:
        result = asdict(self)
        candidates = []
        if any(item["mechanical_route_signal"] for item in self.equipment):
            candidates.append("mechanical_pmv")
        if any(item["adaptive_route_signal"] for item in self.opening_controls):
            candidates.append("natural_adaptive")
        result["comfort_route_candidates"] = candidates
        result["comfort_route_status"] = (
            "candidate_only" if candidates else "undetermined"
        )
        return result


class ZoneInventory:
    def __init__(self, model: ParsedIDF, name_checks: NameChecks) -> None:
        self.model = model
        self.idf = model.idf
        self.name_checks = name_checks
        self.zones = {
            zone.name.upper(): ZoneConfiguration(self._object(zone))
            for zone in self.idf.all_of_type(Zone).values()
            if zone.name.upper() not in name_checks.ambiguous_zones
        }

    def _object(self, obj: IDFBaseModel, *fields: str) -> dict[str, Any]:
        return {
            "location": asdict(self.model.location(obj)),
            "fields": obj.model_dump(
                mode="json", include=set(fields), exclude_none=True
            ),
        }

    def _target_zones(
        self, name: str, *, space_targets: bool = False
    ) -> list[ZoneConfiguration]:
        key = name.upper()
        if key in self.name_checks.ambiguous_zones:
            return []
        if zone := self.zones.get(key):
            return [zone]
        group = self.model.get("ZoneList", name)
        space = self.model.get("Space", name) if space_targets else None
        spaces = self.model.get("SpaceList", name) if space_targets else None
        zone_names: list[str] = []
        if isinstance(group, ZoneList):
            zone_names = [entry.zone_name for entry in group.zones or ()]
        elif isinstance(space, Space):
            zone_names = [space.zone_name]
        elif isinstance(spaces, SpaceList):
            for entry in spaces.spaces or ():
                target = self.model.get("Space", entry.space_name)
                if isinstance(target, Space):
                    zone_names.append(target.zone_name)
        return [
            self.zones[key]
            for key in dict.fromkeys(name.upper() for name in zone_names)
            if key in self.zones
        ]

    def _selected(
        self, kind: str, name: str | None, *fields: str
    ) -> dict[str, Any] | None:
        obj = self.model.get(kind, name)
        return self._object(obj, *fields) if obj is not None else None

    def _people_and_thermostats(self) -> None:
        for space in self.idf.all_of_type(Space).values():
            for zone in self._target_zones(space.zone_name):
                zone.spaces.append(self._object(space, "zone_name", "space_type"))
        for people in self.idf.all_of_type(People).values():
            item = self._object(people, *_PEOPLE_FIELDS)
            item["schedules"] = [
                asdict(self.model.location(obj))
                for obj in people.referenced()
                if obj.idf_object_type().startswith("Schedule:")
            ]
            for zone in self._target_zones(
                people.zone_or_zonelist_or_space_or_spacelist_name, space_targets=True
            ):
                zone.people.append(item)
        for control in self.idf.all_of_type(ZoneControlThermostat).values():
            item = self._object(
                control, "zone_or_zonelist_name", "control_type_schedule_name"
            )
            item["setpoint_objects"] = [
                self._object(
                    target,
                    "heating_setpoint_temperature_schedule_name",
                    "cooling_setpoint_temperature_schedule_name",
                    "setpoint_temperature_schedule_name",
                )
                for target in self.name_checks.thermostat_setpoints[id(control)]
            ]
            for zone in self._target_zones(control.zone_or_zonelist_name):
                zone.thermostats.append(item)

    def _zone_equipment(self) -> None:
        for ideal in self.idf.all_of_type(HVACTemplateZoneIdealLoadsAirSystem).values():
            item = self._object(
                ideal,
                "zone_name",
                "template_thermostat_name",
                "system_availability_schedule_name",
                "heating_availability_schedule_name",
                "cooling_availability_schedule_name",
            )
            item["thermostat"] = self._selected(
                "HVACTemplate:Thermostat",
                ideal.template_thermostat_name,
                "heating_setpoint_schedule_name",
                "constant_heating_setpoint",
                "cooling_setpoint_schedule_name",
                "constant_cooling_setpoint",
            )
            item["mechanical_route_signal"] = True
            for zone in self._target_zones(ideal.zone_name):
                zone.equipment.append(item)
        for connection in self.idf.all_of_type(ZoneHVACEquipmentConnections).values():
            equipment = self.model.get(
                "ZoneHVAC:EquipmentList",
                connection.zone_conditioning_equipment_list_name,
            )
            item = self._object(
                connection,
                "zone_name",
                "zone_conditioning_equipment_list_name",
                "zone_air_inlet_node_or_nodelist_name",
                "zone_return_air_node_or_nodelist_name",
            )
            item["equipment_list"] = None
            item["mechanical_route_signal"] = False
            if isinstance(equipment, ZoneHVACEquipmentList):
                entries = [
                    self._equipment_item(
                        entry.zone_equipment_object_type, entry.zone_equipment_name
                    )
                    for entry in equipment.equipment or ()
                ]
                item["equipment_list"] = self._object(
                    equipment, "load_distribution_scheme"
                )
                item["equipment_list"]["entries"] = entries
                item["mechanical_route_signal"] = any(
                    entry["mechanical_route_signal"] for entry in entries
                )
            for zone in self._target_zones(connection.zone_name):
                zone.equipment.append(item)

    def _equipment_item(self, kind: str, name: str) -> dict[str, Any]:
        obj = self.model.get(kind, name)
        item = {
            "object_type": kind,
            "object_name": name,
            "resolved": obj is not None,
            "object": self._object(
                obj,
                "availability_schedule_name",
                "design_specification_outdoor_air_object_name",
            )
            if obj is not None
            else None,
            "mechanical_route_signal": kind == _IDEAL_LOADS and obj is not None,
        }
        if isinstance(obj, ZoneHVACAirDistributionUnit):
            terminal = self._selected(
                obj.air_terminal_object_type,
                obj.air_terminal_name,
                "availability_schedule_name",
                "air_inlet_node_name",
                "air_outlet_node_name",
                "reheat_coil_object_type",
                "reheat_coil_name",
            )
            item.update(
                terminal_reference={
                    "object_type": obj.air_terminal_object_type,
                    "object_name": obj.air_terminal_name,
                },
                terminal=terminal,
                mechanical_route_signal=terminal is not None
                and obj.air_terminal_object_type in _TERMINALS,
            )
        return item

    def _outdoor_air(self) -> None:
        for sizing in self.idf.all_of_type(SizingZone).values():
            item = self._object(
                sizing,
                "zone_or_zonelist_name",
                "design_specification_outdoor_air_object_name",
            )
            item["specification"] = self._outdoor_air_specification(
                sizing.design_specification_outdoor_air_object_name
            )
            for zone in self._target_zones(sizing.zone_or_zonelist_name):
                zone.outdoor_air_design.append(item)
        outdoor_controllers: dict[str, list[dict[str, Any]]] = {}
        for controller in self.idf.all_of_type(ControllerOutdoorAir).values():
            if name := controller.mechanical_ventilation_controller_name:
                outdoor_controllers.setdefault(name.casefold(), []).append(
                    self._object(controller, "mechanical_ventilation_controller_name")
                )
        for ventilation in self.idf.all_of_type(
            ControllerMechanicalVentilation
        ).values():
            for group in ventilation.zone_specifications or ():
                item = self._object(
                    ventilation,
                    "availability_schedule_name",
                    "system_outdoor_air_method",
                    "demand_controlled_ventilation",
                )
                item["zone_group"] = group.model_dump(mode="json", exclude_none=True)
                item["outdoor_air_controllers"] = outdoor_controllers.get(
                    ventilation.name.casefold(), []
                )
                item["specification"] = self._outdoor_air_specification(
                    group.design_specification_outdoor_air_object_name
                )
                for zone in self._target_zones(group.zone_or_zonelist_name):
                    zone.mechanical_ventilation.append(item)

    def _outdoor_air_specification(self, name: str | None) -> dict[str, Any] | None:
        return self._selected(
            "DesignSpecification:OutdoorAir", name, *_OUTDOOR_AIR_FIELDS
        ) or self._selected("DesignSpecification:OutdoorAir:SpaceList", name)

    def _openings(self) -> None:
        controls: dict[str, list[AirflowNetworkMultiZoneZone]] = {}
        for control in self.idf.all_of_type(AirflowNetworkMultiZoneZone).values():
            controls.setdefault(control.zone_name.casefold(), []).append(control)
        for fenestration in self.idf.all_of_type(FenestrationSurfaceDetailed).values():
            parent = self.model.get(
                "BuildingSurface:Detailed", fenestration.building_surface_name
            )
            if (
                isinstance(parent, BuildingSurfaceDetailed)
                and parent.outside_boundary_condition == "Outdoors"
            ):
                item = self._object(
                    fenestration, "surface_type", "building_surface_name"
                )
                item["parent"] = self._object(
                    parent, "zone_name", "outside_boundary_condition"
                )
                for zone in self._target_zones(parent.zone_name):
                    zone.exterior_fenestration.append(item)
        network = next(
            iter(self.idf.all_of_type(AirflowNetworkSimulationControl).values()), None
        )
        active_network = network is not None and network.airflownetwork_control not in (
            None,
            "",
            "NoMultizoneOrDistribution",
        )
        for opening in self.idf.all_of_type(AirflowNetworkMultiZoneSurface).values():
            item = self._afn_opening(opening)
            if item is None:
                continue
            zone_name, config = item
            zone_controls = controls.get(zone_name.casefold(), [])
            config["network_control"] = (
                self._object(network, "airflownetwork_control")
                if network is not None
                else None
            )
            config["zone_controls"] = [
                self._zone_opening_control(control) for control in zone_controls
            ]
            config["occupant_control"] = self._selected(
                "AirflowNetwork:OccupantVentilationControl",
                opening.occupant_ventilation_control_name,
            )
            zone_level_control = opening.ventilation_control_mode in (
                "ZoneLevel",
                None,
                "",
            )
            config["adaptive_route_signal"] = (
                config["adaptive_route_signal"]
                and active_network
                and (
                    not zone_level_control
                    or any(
                        control.ventilation_control_mode not in ("NoVent", None, "")
                        for control in zone_controls
                    )
                )
            )
            for zone in self._target_zones(zone_name):
                zone.opening_controls.append(config)
        for opening in self.idf.all_of_type(
            ZoneVentilationWindandStackOpenArea
        ).values():
            item = self._object(
                opening,
                "zone_or_space_name",
                "opening_area",
                "opening_area_fraction_schedule_name",
            )
            item["adaptive_route_signal"] = (opening.opening_area or 0) > 0
            for zone in self._target_zones(
                opening.zone_or_space_name, space_targets=True
            ):
                zone.opening_controls.append(item)

    def _zone_opening_control(
        self, control: AirflowNetworkMultiZoneZone
    ) -> dict[str, Any]:
        item = self._object(
            control,
            "zone_name",
            "ventilation_control_mode",
            "occupant_ventilation_control_name",
        )
        item["occupant_control"] = self._selected(
            "AirflowNetwork:OccupantVentilationControl",
            control.occupant_ventilation_control_name,
        )
        return item

    def _afn_opening(
        self, opening: AirflowNetworkMultiZoneSurface
    ) -> tuple[str, dict[str, Any]] | None:
        surface = self.model.get("FenestrationSurface:Detailed", opening.surface_name)
        if not isinstance(surface, FenestrationSurfaceDetailed):
            return None
        parent = self.model.get(
            "BuildingSurface:Detailed", surface.building_surface_name
        )
        if (
            not isinstance(parent, BuildingSurfaceDetailed)
            or parent.outside_boundary_condition != "Outdoors"
        ):
            return None
        components = [
            self._object(obj)
            for obj in opening.referenced()
            if obj.idf_object_type() in _OPENING_TYPES
        ]
        if not components:
            return None
        item = self._object(
            opening,
            "surface_name",
            "leakage_component_name",
            "ventilation_control_mode",
            "window_door_opening_factor_or_crack_factor",
            "occupant_ventilation_control_name",
            "venting_availability_schedule_name",
        )
        item["opening_components"] = components
        item["parent"] = self._object(parent, "zone_name", "outside_boundary_condition")
        item["adaptive_route_signal"] = (
            opening.ventilation_control_mode != "NoVent"
            and opening.window_door_opening_factor_or_crack_factor != 0
        )
        return parent.zone_name, item

    def air_loop_configurations(self) -> list[dict[str, Any]]:
        return [
            self._object(
                loop,
                "controller_list_name",
                "availability_manager_list_name",
                "branch_list_name",
                "supply_side_inlet_node_name",
                "supply_side_outlet_node_names",
                "demand_side_inlet_node_names",
            )
            for loop in self.idf.all_of_type(AirLoopHVAC).values()
        ]

    def report(self) -> list[dict[str, Any]]:
        self._people_and_thermostats()
        self._zone_equipment()
        self._outdoor_air()
        self._openings()
        return [zone.report() for zone in self.zones.values()]


def unsupported_topology(model: ParsedIDF) -> list[Finding]:
    """Only flag unmodeled equipment/topology relevant to zone attribution."""
    findings = []
    for obj in model.idf:
        kind = obj.idf_object_type()
        if kind.startswith(
            (
                "ZoneHVAC:",
                "HVACTemplate:",
                "AirTerminal:",
                "AirflowNetwork:Distribution:",
            )
        ) and kind not in _SUPPORTED_ZONE_HVAC | _TERMINALS | {
            "HVACTemplate:Thermostat"
        }:
            findings.append(
                Finding(
                    "support_boundary",
                    "topology_not_evaluated",
                    model.location(obj),
                    "This object is recognized by the 26.1 schema but has no first-release zone evaluation mapping.",
                    "Provide a verified object-to-zone and control mapping, or an evaluator supporting this topology.",
                )
            )
    return findings
