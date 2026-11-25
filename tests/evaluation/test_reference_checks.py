import subprocess
import sys
from pathlib import Path

import pytest

from src.evaluation.static_idf import evaluate

_THERMOSTAT = """Version,26.1;
Zone,Z;
Schedule:Constant,ControlType,,2;
Schedule:Constant,Setpoint,,21;
ThermostatSetpoint:SingleHeating,HeatOnly,Setpoint;
"""
_CONSTRUCTION = """Version,26.1;
Material,M,Rough,0.1,0.5,800,1000;
"""


def _diagnose(tmp_path: Path, body: str):
    path = tmp_path / "case.idf"
    path.write_text(body, encoding="utf-8")
    return evaluate(path)


def _finding(report, code):
    assert report["status"] == "invalid_input"
    assert report["checks"]["object_names_and_target_resolution"] == "failed"
    issue = next(item for item in report["findings"] if item["code"] == code)
    assert issue["required_material"]
    assert issue["details"]["resolution"] == "blocked"
    return issue


@pytest.mark.parametrize("second", ["probezone", "ProbeZone"])
def test_duplicate_zone_name_blocks_only_conflicted_zone(tmp_path, second):
    report = _diagnose(
        tmp_path,
        f"""Version,26.1;
Zone,ProbeZone;
Zone,{second};
Zone,Independent;
Schedule:Constant,Occ,,1;
People,Ambiguous,pRoBeZoNe,Occ,People,1,,,,,Occ;
People,Known,independent,Occ,People,2,,,,,Occ;
""",
    )
    issue = _finding(report, "duplicate_object_name")
    assert issue["location"]["field"] == "name"
    assert [
        (item["object_name"], item["line"]) for item in issue["details"]["candidates"]
    ] == [
        ("ProbeZone", 2),
        (second, 3),
    ]
    assert {item["object_name"] for item in issue["details"]["affected"]} >= {
        "ProbeZone",
        "Ambiguous",
    }
    assert len(report["zones"]) == 1
    assert report["zones"][0]["zone"]["location"]["object_name"] == "Independent"
    assert report["zones"][0]["people"][0]["location"]["object_name"] == "Known"


def test_single_zone_accepts_mixed_case_reference(tmp_path):
    report = _diagnose(
        tmp_path,
        """Version,26.1;
Zone,ProbeZone;
Schedule:Constant,Occ,,1;
People,P,pRoBeZoNe,Occ,People,1,,,,,Occ;
""",
    )
    assert report["status"] == "static_complete"
    assert report["zones"][0]["people"][0]["location"]["object_name"] == "P"


@pytest.mark.parametrize(
    ("layers", "field"), [("M", "outside_layer"), ("Other,m", "layer_2")]
)
def test_construction_checks_each_material_layer_in_actual_group(
    tmp_path, layers, field
):
    report = _diagnose(
        tmp_path,
        _CONSTRUCTION
        + "Material:NoMass,m,Rough,0.5;\nMaterial:NoMass,Other,Rough,0.4;\n"
        + f"Construction,C,{layers};\n",
    )
    issue = _finding(report, "ambiguous_construction_material")
    assert issue["location"] == {
        "object_type": "Construction",
        "object_name": "C",
        "line": 5,
        "field": field,
    }
    assert issue["details"]["reference_groups"] == ["MaterialName"]
    assert [
        (item["object_type"], item["object_name"], item["line"])
        for item in issue["details"]["candidates"]
    ] == [
        ("Material", "M", 2),
        ("Material:NoMass", "m", 3),
    ]


@pytest.mark.parametrize("additional", ["", "Schedule:Constant,M,,1;\n"])
def test_construction_ignores_same_name_in_unrelated_group(tmp_path, additional):
    report = _diagnose(tmp_path, _CONSTRUCTION + additional + "Construction,C,m;\n")
    assert report["status"] == "static_complete"
    assert report["checks"]["object_names_and_target_resolution"] == "passed"


@pytest.mark.parametrize(
    ("type_suffix", "target", "additional", "code", "num_candidates"),
    [
        ("SingleCooling", "HeatOnly", "", "thermostat_target_type_mismatch", 1),
        ("SingleCooling", "Absent", "", "thermostat_target_missing", 0),
        (
            "SingleHeating",
            "HeatOnly",
            "ThermostatSetpoint:SingleCooling,heatonly,Setpoint;\n",
            "thermostat_target_ambiguous",
            2,
        ),
        (
            "SingleHeating",
            "HEATONLY",
            "ThermostatSetpoint:SingleHeating,HeatOnly,Setpoint;\n",
            "thermostat_target_ambiguous",
            2,
        ),
    ],
)
def test_thermostat_pair_is_checked_before_becoming_zone_fact(
    tmp_path, type_suffix, target, additional, code, num_candidates
):
    report = _diagnose(
        tmp_path,
        _THERMOSTAT
        + additional
        + f"ZoneControl:Thermostat,Ctrl,Z,ControlType,ThermostatSetpoint:{type_suffix},{target};\n",
    )
    issue = _finding(report, code)
    assert issue["location"]["object_name"] == "Ctrl"
    details = issue["details"]
    assert details["paired_fields"] == ["control_1_object_type", "control_1_name"]
    assert [location["field"] for location in details["pair_locations"]] == details[
        "paired_fields"
    ]
    assert len(details["candidates"]) == num_candidates
    assert {item["object_name"] for item in details["affected"]} == {"Ctrl", "Z"}
    assert report["zones"][0]["thermostats"][0]["setpoint_objects"] == []


def test_mixed_model_preserves_valid_pairs_on_same_control_and_other_zone(tmp_path):
    report = _diagnose(
        tmp_path,
        _THERMOSTAT
        + """Zone,Independent;
ThermostatSetpoint:SingleCooling,CoolOnly,Setpoint;
ZoneControl:Thermostat,Ctrl,Z,ControlType,ThermostatSetpoint:SingleHeating,hEAToNLY,ThermostatSetpoint:SingleCooling,HeatOnly;
ZoneControl:Thermostat,Good,Independent,ControlType,ThermostatSetpoint:SingleCooling,coolonly;
""",
    )
    issue = _finding(report, "thermostat_target_type_mismatch")
    assert issue["details"]["paired_fields"] == [
        "control_2_object_type",
        "control_2_name",
    ]
    assert report["checks"]["zone_configuration"] == "partial"
    assert [
        [
            target["location"]["object_name"]
            for target in zone["thermostats"][0]["setpoint_objects"]
        ]
        for zone in report["zones"]
    ] == [["HeatOnly"], ["CoolOnly"]]


def test_cli_summary_includes_blocked_scope_and_both_declarations(tmp_path):
    path = tmp_path / "case.idf"
    path.write_text(
        "Version,26.1;\nZone,ProbeZone;\nZone,probezone;\n", encoding="utf-8"
    )
    result = subprocess.run(
        [sys.executable, "-m", "src.evaluation.static_idf", path.name],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    for text in (
        "duplicate_object_name",
        "line 2 / Zone / ProbeZone / name",
        "line 3 / Zone / probezone / name",
        "Affected:",
        "To unblock:",
    ):
        assert text in result.stderr
