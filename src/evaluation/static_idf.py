"""Diagnose one EnergyPlus 26.1 IDF without weather, simulation, or model generation."""

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version as distribution_version
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from src.evaluation.idf_input import (
    SUPPORTED_VERSIONS,
    Finding,
    Location,
    ParsedIDF,
    blocks,
    version_check,
)
from src.evaluation.reference_checks import check_names_and_references
from src.evaluation.zone_config import ZoneInventory, unsupported_topology
from src.modeling.validation import (
    completeness_issues,
    fenestration_issues,
    interzone_issues,
)

_EXIT_CODES = {
    "static_complete": 0,
    "invalid_input": 2,
    "unsupported_version": 3,
    "limited_scope": 4,
}


def _evidence_requirements() -> dict[str, Any]:
    return {
        "weather": {
            "status": "missing",
            "required_material": "Supply the matching EPW with SHA-256, location and simulation calendar.",
        },
        "run_outputs": {
            "status": "missing",
            "required_material": (
                "Supply a run manifest linking source and executed IDF hashes, EPW and engine build; "
                "same-run ERR/END, SQL or ESO, output dictionaries and zone-time coverage."
            ),
        },
        "project": {
            "status": "missing",
            "required_material": (
                "Supply approved comfort method and occupancy periods, representative occupants, "
                "controls and clothing assumptions, service limits and the required energy boundary."
            ),
        },
        "mechanical_pmv": {
            "status": "needs_evidence",
            "required_material": (
                "Confirm mechanical operation and occupancy for the same zone intervals. Supply "
                "air and radiant temperatures, humidity, air speed, met, clothing and external work."
            ),
        },
        "natural_adaptive": {
            "status": "needs_evidence",
            "required_material": (
                "Supply occupied operative temperatures and seven preceding complete local weather days. "
                "Confirm occupant control of openings, primary thermal control, clothing freedom, "
                "activity range and mechanical heating/cooling inactivity for the intervals."
            ),
        },
        "energy_savings": {
            "status": "needs_evidence",
            "required_material": (
                "Supply comparable baseline/candidate runs, same-window energy meters by carrier "
                "and evidence of maintained comfort and required services."
            ),
        },
    }


def evaluate(path: Path) -> dict[str, Any]:
    """Build the versioned result from the same bytes used for input identity."""
    input_data: dict[str, Any] = {
        "path": str(path.absolute()),
        "sha256": None,
        "bytes": None,
        "version": None,
    }
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "input": input_data,
        "scope": {
            "idfpy_version": distribution_version("idfpy"),
            "supported_versions": list(SUPPORTED_VERSIONS),
            "evaluation": "idf_static",
            "limits": [
                "UTF-8 IDF text and pinned idfpy schema fields and declared reference groups.",
                "Additional name resolution covers Zone declarations, Construction material layers and ZoneControl:Thermostat setpoint pairs.",
                "Main-library geometry checks are advisory.",
                "Only mapped zone equipment, control and opening configurations inform candidate routes.",
                "No EnergyPlus execution, template expansion, full HVAC node graph, schedule expansion or physical validation.",
            ],
        },
        "checks": {
            "input_snapshot": "not_run",
            "version": "not_run",
            "idfpy_strict_schema": "not_run",
            "idfpy_declared_references": "not_run",
            "object_names_and_target_resolution": "not_run",
            "modeling_geometry": "not_run",
            "builder_completeness": "not_run",
            "zone_configuration": "not_run",
        },
        "findings": [],
        "objects": {},
        "unvalidated_objects": [],
        "zones": [],
        "air_loops": [],
        "output_request_objects": {},
        "evidence": _evidence_requirements(),
        "determinations": {
            "operating_mode": "unknown",
            "comfort_compliance": "not_evaluated",
            "required_services": "not_evaluated",
            "energy_use": "not_evaluated",
            "energy_savings": "not_evaluated",
        },
    }
    try:
        raw = path.read_bytes()
    except OSError as error:
        result["checks"]["input_snapshot"] = "failed"
        return _complete(
            result,
            "invalid_input",
            [
                Finding(
                    "input_error",
                    "input_unreadable",
                    Location(),
                    str(error),
                    "Provide a readable IDF file path.",
                )
            ],
        )

    input_data.update(sha256=sha256(raw).hexdigest(), bytes=len(raw))
    result["checks"]["input_snapshot"] = "passed"
    try:
        items = list(blocks(raw.decode("utf-8-sig")))
    except UnicodeDecodeError as error:
        return _complete(
            result,
            "invalid_input",
            [
                Finding(
                    "input_error",
                    "input_encoding",
                    Location(line=raw[: error.start].count(b"\n") + 1),
                    "The input contains bytes outside the supported UTF-8 encoding.",
                    "Provide a UTF-8 copy and record the change to the source identity.",
                )
            ],
        )
    if not items:
        return _complete(
            result,
            "invalid_input",
            [
                Finding(
                    "input_error",
                    "input_empty",
                    Location(),
                    "The file contains no IDF objects.",
                    "Supply an EnergyPlus 26.1 IDF containing model objects.",
                )
            ],
        )
    declared, version_issue = version_check(items)
    input_data["version"] = declared
    if version_issue:
        result["checks"]["version"] = "failed"
        status = (
            "invalid_input"
            if version_issue.category == "input_error"
            else "unsupported_version"
        )
        return _complete(result, status, [version_issue])

    result["checks"]["version"] = "passed"
    model = ParsedIDF(items, input_data["path"])
    return _collect(result, model)


def _collect(result: dict[str, Any], model: ParsedIDF) -> dict[str, Any]:
    checks = result["checks"]
    references = list(model.reference_findings())
    name_checks = check_names_and_references(model)
    checks["idfpy_strict_schema"] = "failed" if model.findings else "passed"
    checks["idfpy_declared_references"] = (
        "partial" if model.partial_objects else "failed" if references else "passed"
    )
    checks["object_names_and_target_resolution"] = (
        "failed"
        if name_checks.findings
        else "partial"
        if model.partial_objects
        else "passed"
    )
    counts = Counter(obj.idf_object_type() for obj in model.idf)
    result["objects"] = dict(sorted(counts.items()))
    result["unvalidated_objects"] = model.partial_objects
    result["output_request_objects"] = {
        kind: count
        for kind, count in result["objects"].items()
        if kind.startswith("Output:")
    }
    geometry_findings = _geometry_findings(model)
    checks["modeling_geometry"] = "partial" if model.partial_objects else "executed"
    completeness = [
        Finding(
            "builder_requirement",
            "builder_completeness",
            Location(issue.object_type, issue.object_name, field=issue.field),
            issue.message,
            "If using the model builder, supply zones and detailed building surfaces or document the alternative representation.",
        )
        for issue in completeness_issues(model.idf)
    ]
    checks["builder_completeness"] = "partial" if model.partial_objects else "executed"
    bounded = unsupported_topology(model)
    inventory = ZoneInventory(model, name_checks)
    result["zones"] = inventory.report()
    result["air_loops"] = inventory.air_loop_configurations()
    checks["zone_configuration"] = (
        "partial"
        if model.partial_objects or bounded or references or name_checks.findings
        else "executed"
    )
    specific_locations = {finding.location for finding in name_checks.findings}
    findings = (
        name_checks.findings
        + model.findings
        + [
            finding
            for finding in references
            if finding.location not in specific_locations
        ]
        + geometry_findings
        + completeness
        + bounded
    )
    if any(finding.category == "input_error" for finding in findings):
        status = "invalid_input"
    elif bounded:
        status = "limited_scope"
    else:
        status = "static_complete"
    return _complete(result, status, findings)


def _geometry_findings(model: ParsedIDF) -> list[Finding]:
    findings = []
    for checker in (fenestration_issues, interzone_issues):
        for issue in checker(model.idf):
            obj = (
                model.get(issue.object_type, issue.object_name)
                if issue.object_type
                else None
            )
            location = (
                model.location(obj, issue.field)
                if obj is not None
                else Location(issue.object_type, issue.object_name, field=issue.field)
            )
            findings.append(
                Finding(
                    "modeling_advisory",
                    checker.__name__,
                    location,
                    issue.message,
                    "Review surface coordinates, orientation and the paired surface in the source.",
                )
            )
    return findings


def _complete(
    result: dict[str, Any], status: str, findings: list[Finding]
) -> dict[str, Any]:
    result["status"] = status
    result["findings"] = [asdict(finding) for finding in findings]
    return result


def _write_report(path: Path, input_path: Path, report: str) -> None:
    if path.resolve() == input_path.resolve() or (
        path.exists() and input_path.exists() and path.samefile(input_path)
    ):
        raise ValueError("JSON output must not refer to the input IDF.")
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w", dir=path.parent, encoding="utf-8", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(report)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _format_location(location: dict[str, Any]) -> str:
    return " / ".join(
        str(part)
        for part in (
            f"line {location['line']}" if location["line"] else None,
            location["object_type"],
            location["object_name"],
            location["field"],
        )
        if part is not None
    )


def _print_summary(result: dict[str, Any]) -> None:
    counts = Counter(item["category"] for item in result["findings"])
    routes = Counter(
        route for zone in result["zones"] for route in zone["comfort_route_candidates"]
    )
    print(
        f"{result['status']} | EnergyPlus {result['input']['version'] or 'undeclared'} | "
        f"{len(result['zones'])} zones | {sum(counts.values())} findings | "
        "weather, run outputs and project evidence missing",
        file=sys.stderr,
    )
    if routes:
        print(
            "Candidate routes by zone: "
            + ", ".join(f"{name} ({count})" for name, count in sorted(routes.items())),
            file=sys.stderr,
        )
    for finding in result["findings"][:5]:
        where = _format_location(finding["location"])
        print(
            f"- {finding['code']} [{where or 'input'}] {finding['reason']}",
            file=sys.stderr,
        )
        details = finding["details"]
        if fields := details.get("paired_fields"):
            print("  Paired fields: " + ", ".join(fields), file=sys.stderr)
        if candidates := details.get("candidates"):
            print(
                "  Candidates: " + " | ".join(map(_format_location, candidates[:3])),
                file=sys.stderr,
            )
        if affected := details.get("affected"):
            print(
                "  Affected: " + " | ".join(map(_format_location, affected[:3])),
                file=sys.stderr,
            )
            if len(affected) > 3:
                print(
                    f"  See JSON for {len(affected) - 3} more locations.",
                    file=sys.stderr,
                )
            print(f"  To unblock: {finding['required_material']}", file=sys.stderr)
    if len(result["findings"]) > 5:
        print("- See JSON for all findings and required materials.", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("idf", type=Path, help="path to the input IDF")
    parser.add_argument(
        "--json", type=Path, help="write JSON to this path instead of stdout"
    )
    args = parser.parse_args(argv)
    result = evaluate(args.idf)
    try:
        report = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.json is None:
            sys.stdout.write(report)
        else:
            _write_report(args.json, args.idf, report)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Cannot write JSON: {error}", file=sys.stderr)
        return 1
    _print_summary(result)
    return _EXIT_CODES[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
