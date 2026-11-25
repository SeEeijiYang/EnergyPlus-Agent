import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from src.evaluation.static_idf import evaluate

_MODEL = """Version,26.1;
Zone,North Room;
Zone,South Room;
Schedule:Constant,Always,,1;
People,Occupants,North Room,Always,People,2,,,,,Always;
"""


def _write_model(tmp_path: Path, text: str = _MODEL) -> Path:
    path = tmp_path / "新模型 空间.idf"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("version", ["26.1", "26.1.0"])
def test_static_diagnosis_without_weather_or_run(tmp_path, version):
    path = _write_model(tmp_path, _MODEL.replace("26.1", version))
    before = path.read_bytes()
    report = evaluate(path)
    assert report["status"] == "static_complete"
    assert report["schema_version"] == "1.0"
    assert report["input"]["sha256"] == sha256(before).hexdigest()
    assert report["input"]["version"] == version
    assert report["checks"]["idfpy_declared_references"] == "passed"
    assert report["evidence"]["weather"]["status"] == "missing"
    assert report["evidence"]["run_outputs"]["status"] == "missing"
    assert report["evidence"]["project"]["status"] == "missing"
    assert report["determinations"]["operating_mode"] == "unknown"
    assert report["determinations"]["energy_savings"] == "not_evaluated"
    assert {finding["category"] for finding in report["findings"]} == {
        "builder_requirement"
    }
    assert report["zones"][0]["people"][0]["schedules"][0]["object_name"] == "Always"
    assert report["zones"][0]["comfort_route_candidates"] == []
    assert path.read_bytes() == before


def test_located_bad_reference_preserves_other_fields(tmp_path):
    path = _write_model(
        tmp_path, _MODEL.replace("North Room,Always", "North Room,Missing")
    )
    report = evaluate(path)
    assert report["status"] == "invalid_input"
    issue = next(
        finding
        for finding in report["findings"]
        if finding["code"] == "reference_missing"
    )
    assert issue["location"] == {
        "object_type": "People",
        "object_name": "Occupants",
        "line": 5,
        "field": "number_of_people_schedule_name",
    }
    assert issue["details"]["referenced_name"] == "Missing"
    assert issue["required_material"]
    assert report["zones"][0]["people"][0]["fields"]["number_of_people"] == 2
    assert report["zones"][1]["zone"]["location"]["object_name"] == "South Room"


def test_local_field_failure_retains_raw_and_parsed_objects(tmp_path):
    path = _write_model(tmp_path, _MODEL.replace("People,2,", "People,not numeric,"))
    report = evaluate(path)
    assert report["status"] == "invalid_input"
    assert report["checks"]["idfpy_strict_schema"] == "failed"
    assert report["checks"]["idfpy_declared_references"] == "partial"
    issue = next(
        finding for finding in report["findings"] if finding["code"] == "invalid_field"
    )
    assert issue["location"]["field"] == "number_of_people"
    assert issue["location"]["object_name"] == "Occupants"
    partial = report["unvalidated_objects"][0]
    assert partial["raw_field_values"]["activity_level_schedule_name"] == "Always"
    assert partial["status"] == "unvalidated"
    assert len(report["zones"]) == 2
    assert report["objects"]["Schedule:Constant"] == 1


def test_reference_to_rejected_provider_is_provisional(tmp_path):
    text = _MODEL.replace("Always,,1;", "Always,,not numeric;")
    report = evaluate(_write_model(tmp_path, text))
    assert report["status"] == "invalid_input"
    issues = [
        issue
        for issue in report["findings"]
        if issue["code"] == "reference_target_unvalidated"
    ]
    assert {issue["location"]["field"] for issue in issues} == {
        "activity_level_schedule_name",
        "number_of_people_schedule_name",
    }
    assert report["zones"][0]["people"][0]["fields"]["number_of_people"] == 2


def test_version_outside_scope_skips_pinned_schema(tmp_path):
    report = evaluate(_write_model(tmp_path, _MODEL.replace("26.1", "25.2")))
    assert report["status"] == "unsupported_version"
    assert report["input"]["version"] == "25.2"
    assert report["checks"]["idfpy_strict_schema"] == "not_run"
    assert report["findings"][0]["location"]["line"] == 1
    assert report["findings"][0]["location"]["field"] == "version_identifier"


def test_legal_unmapped_topology_is_separate_from_input_errors(tmp_path):
    path = _write_model(
        tmp_path,
        _MODEL
        + """
ZoneHVAC:Baseboard:Convective:Electric,Electric Heater,Always,HeatingDesignCapacity,1000,,,1;
""",
    )
    report = evaluate(path)
    assert report["status"] == "limited_scope"
    assert report["checks"]["idfpy_strict_schema"] == "passed"
    assert report["checks"]["idfpy_declared_references"] == "passed"
    finding = next(
        item for item in report["findings"] if item["category"] == "support_boundary"
    )
    assert finding["location"]["object_name"] == "Electric Heater"
    assert finding["location"]["line"] == 7
    assert not any(item["category"] == "input_error" for item in report["findings"])
    assert len(report["zones"]) == 2


def test_space_list_people_and_opening_map_to_zones(tmp_path):
    model = """Version,26.1;
Zone,North Room;
Zone,South Room;
Space,Desk A,North Room;
Space,Desk B,South Room;
SpaceList,Workspaces,Desk A,Desk B;
Schedule:Constant,Always,,1;
People,Occupants,Workspaces,Always,People,2,,,,,Always;
ZoneVentilation:WindandStackOpenArea,Passive Opening,Desk B,0.4,Always;
"""
    report = evaluate(_write_model(tmp_path, model))
    assert report["status"] == "static_complete"
    north, south = report["zones"]
    assert north["spaces"][0]["location"]["object_name"] == "Desk A"
    assert south["spaces"][0]["location"]["object_name"] == "Desk B"
    assert north["people"][0]["location"] == south["people"][0]["location"]
    assert north["comfort_route_candidates"] == []
    assert south["comfort_route_candidates"] == ["natural_adaptive"]


def test_missing_terminal_semicolon_is_located(tmp_path):
    report = evaluate(_write_model(tmp_path, _MODEL + "Zone,Incomplete"))
    assert report["status"] == "invalid_input"
    assert report["findings"][0]["code"] == "invalid_object_syntax"
    assert report["findings"][0]["location"]["object_name"] == "Incomplete"
    assert len(report["zones"]) == 2


def test_cli_separates_json_and_summary_and_protects_input_alias(tmp_path):
    path = _write_model(tmp_path)
    before = path.read_bytes()
    command = [sys.executable, "-m", "src.evaluation.static_idf", path.name]
    run = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert run.returncode == 0
    assert json.loads(run.stdout)["input"]["sha256"] == sha256(before).hexdigest()
    assert "weather, run outputs and project evidence missing" in run.stderr
    alias = tmp_path / "alias.json"
    alias.hardlink_to(path)
    run = subprocess.run(
        [*command, "--json", alias.name],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 1
    assert "JSON output must not refer to the input IDF" in run.stderr
    assert path.read_bytes() == alias.read_bytes() == before
