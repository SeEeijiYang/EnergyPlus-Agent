"""Check names and resolve selected links that idfpy only checks for existence."""

from dataclasses import asdict, dataclass, field
from typing import Any

from idfpy import IDFBaseModel
from idfpy.models import Construction, ZoneControlThermostat, ZoneList
from idfpy.models._ref_meta import REF_CONSUMERS

from src.evaluation.idf_input import Finding, Location, ParsedIDF, ProviderDeclaration

_MATERIAL_FIELDS = REF_CONSUMERS[Construction.__name__]
_THERMOSTAT_FIELDS = REF_CONSUMERS[ZoneControlThermostat.__name__]


@dataclass(slots=True)
class NameChecks:
    findings: list[Finding] = field(default_factory=list)
    ambiguous_zones: set[str] = field(default_factory=set)
    thermostat_setpoints: dict[int, list[IDFBaseModel]] = field(default_factory=dict)


def _blocked(
    code: str,
    source: Location,
    reason: str,
    required_material: str,
    affected: list[Location],
    candidates: list[ProviderDeclaration],
    **details: Any,
) -> Finding:
    return Finding(
        "input_error",
        code,
        source,
        reason,
        required_material,
        {
            "resolution": "blocked",
            "affected": [asdict(location) for location in dict.fromkeys(affected)],
            "candidates": [
                {**asdict(candidate.location), "parsed": candidate.target is not None}
                for candidate in candidates
            ],
            **details,
        },
    )


def _check_zone_names(model: ParsedIDF, checks: NameChecks) -> None:
    for group in model.provider_declarations("ZoneNames"):
        candidates = [d for d in group if d.location.object_type == "Zone"]
        if len(candidates) < 2:
            continue
        first, conflicting = candidates[:2]
        name = first.location.object_name or ""
        checks.ambiguous_zones.add(name.upper())
        declarations = [item.location for item in candidates]
        consumers = [
            model.location(consumer)
            for candidate in candidates
            if candidate.target is not None
            for consumer in candidate.target.referencing()
        ]
        spellings = ", ".join(repr(item.object_name) for item in declarations)
        checks.findings.append(
            _blocked(
                "duplicate_object_name",
                conflicting.location,
                f"Zone declarations {spellings} share one case-insensitive name. Zone attribution is blocked.",
                "Give the zones unique names and update their references, or remove the duplicate declaration.",
                declarations + consumers,
                candidates,
                zone_name=name,
            )
        )


def _check_materials(model: ParsedIDF, checks: NameChecks) -> None:
    for construction in model.idf.all_of_type(Construction).values():
        for layer, groups in _MATERIAL_FIELDS.items():
            name = getattr(construction, layer, None)
            if not name:
                continue
            candidates = model.providers(name, groups)
            if len(candidates) < 2:
                continue
            source = model.location(construction, layer)
            checks.findings.append(
                _blocked(
                    "ambiguous_construction_material",
                    source,
                    f"Material name {name!r} has {len(candidates)} candidates in the layer's allowed reference groups.",
                    "Give the materials distinct names and select exactly one allowed material for this layer.",
                    [
                        source,
                        *(
                            model.location(consumer)
                            for consumer in construction.referencing()
                        ),
                    ],
                    candidates,
                    referenced_name=name,
                    reference_groups=groups,
                )
            )


def _thermostat_scope(
    model: ParsedIDF, control: ZoneControlThermostat
) -> list[Location]:
    affected = []
    for zone in model.providers(
        control.zone_or_zonelist_name, _THERMOSTAT_FIELDS["zone_or_zonelist_name"]
    ):
        affected.append(zone.location)
        if isinstance(zone.target, ZoneList):
            for entry in zone.target.zones or ():
                affected.extend(
                    member.location
                    for member in model.providers(entry.zone_name, ("ZoneNames",))
                )
    return affected


def _check_thermostats(model: ParsedIDF, checks: NameChecks) -> None:
    for control in model.idf.all_of_type(ZoneControlThermostat).values():
        resolved = {}
        for slot in range(1, 5):
            type_field, name_field = (
                f"control_{slot}_object_type",
                f"control_{slot}_name",
            )
            kind, name = getattr(control, type_field), getattr(control, name_field)
            if not kind and not name:
                continue
            groups = _THERMOSTAT_FIELDS[name_field]
            candidates = model.providers(name, groups) if name else []
            code, reason = _thermostat_problem(kind, name, candidates)
            if code:
                source = model.location(control, name_field)
                checks.findings.append(
                    _blocked(
                        code,
                        source,
                        reason,
                        "Provide both fields and make the name unique in the setpoint group with the declared type.",
                        [source, *_thermostat_scope(model, control)],
                        candidates,
                        paired_fields=[type_field, name_field],
                        pair_locations=[
                            asdict(model.location(control, key))
                            for key in (type_field, name_field)
                        ],
                        declared_type=kind,
                        referenced_name=name,
                        reference_groups=groups,
                    )
                )
            else:
                target = candidates[0].target
                if target is not None:
                    resolved[id(target)] = target
        checks.thermostat_setpoints[id(control)] = list(resolved.values())


def _thermostat_problem(
    kind: str | None, name: str | None, candidates: list[ProviderDeclaration]
) -> tuple[str | None, str]:
    if not kind or not name:
        return (
            "thermostat_target_missing",
            "The thermostat control slot needs a type and a target name.",
        )
    if not candidates:
        return (
            "thermostat_target_missing",
            f"No setpoint named {name!r} exists in the allowed group.",
        )
    if len(candidates) > 1:
        return (
            "thermostat_target_ambiguous",
            f"Setpoint name {name!r} has {len(candidates)} candidates in the allowed group.",
        )
    candidate = candidates[0]
    if candidate.location.object_type != kind:
        return (
            "thermostat_target_type_mismatch",
            f"The declared type {kind} differs from the type of {name!r}, {candidate.location.object_type}.",
        )
    if candidate.target is None:
        return (
            "thermostat_target_unvalidated",
            f"Setpoint {name!r} could not be parsed.",
        )
    return None, ""


def check_names_and_references(model: ParsedIDF) -> NameChecks:
    checks = NameChecks()
    _check_zone_names(model, checks)
    _check_materials(model, checks)
    _check_thermostats(model, checks)
    return checks
