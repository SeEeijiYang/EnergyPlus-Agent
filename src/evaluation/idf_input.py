"""Read located IDF blocks and validate each with the project's idfpy schema."""

from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

from idfpy import IDF, IDFBaseModel, RefError
from idfpy.idf import FIELD_ORDER_REGISTRY
from idfpy.models._ref_meta import REF_PROVIDERS
from pydantic import ValidationError

from src.modeling.validation import describe_reference_issue

SUPPORTED_VERSIONS = ("26.1", "26.1.0")
_OBJECT_TYPES = {name.casefold(): name for name in FIELD_ORDER_REGISTRY}


@dataclass(frozen=True, slots=True)
class Location:
    object_type: str | None = None
    object_name: str | None = None
    line: int | None = None
    field: str | None = None


@dataclass(frozen=True, slots=True)
class Finding:
    category: str
    code: str
    location: Location
    reason: str
    required_material: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProviderDeclaration:
    location: Location
    target: IDFBaseModel | None


@dataclass(frozen=True, slots=True)
class Block:
    fields: list[str]
    line: int
    terminated: bool

    @property
    def kind(self) -> str:
        token = self.fields[0]
        return _OBJECT_TYPES.get(token.casefold(), token)


def blocks(text: str) -> Iterator[Block]:
    """Identify IDF object boundaries using idfpy's IDF delimiter rules."""
    chunks: list[str] = []
    first_line = 0
    for number, raw_line in enumerate(text.splitlines(), 1):
        segments = raw_line.partition("!")[0].split(";")
        for index, segment in enumerate(segments):
            if segment := segment.strip():
                if not chunks:
                    first_line = number
                chunks.append(segment)
            if index < len(segments) - 1 and chunks:
                yield Block(_split_fields(chunks), first_line, True)
                chunks = []
    if chunks:
        yield Block(_split_fields(chunks), first_line, False)


def _split_fields(chunks: list[str]) -> list[str]:
    return [value.strip() for value in " ".join(chunks).split(",")]


def version_check(items: list[Block]) -> tuple[str | None, Finding | None]:
    versions = [block for block in items if block.kind == "Version"]
    if not versions:
        return None, Finding(
            "support_boundary",
            "version_missing",
            Location("Version", field="version_identifier"),
            "The input does not declare an EnergyPlus version.",
            "Provide an IDF with an explicit EnergyPlus 26.1 Version object.",
        )
    first = versions[0]
    version = first.fields[1] if len(first.fields) > 1 else ""
    if len(versions) > 1 or not first.terminated or not version:
        line = versions[1].line if len(versions) > 1 else first.line
        return version or None, Finding(
            "input_error",
            "invalid_version_object",
            Location("Version", line=line, field="version_identifier"),
            "Version must appear once with a value and a terminating semicolon.",
            "Supply a single complete Version object using the model's true version.",
        )
    if version not in SUPPORTED_VERSIONS:
        return version, Finding(
            "support_boundary",
            "version_unsupported",
            Location("Version", line=first.line, field="version_identifier"),
            f"The evaluator has no schema checks for the declared version {version!r}.",
            "Provide a model migrated to EnergyPlus 26.1 or use its matching evaluator.",
        )
    return version, None


def _raw_fields(block: Block) -> dict[str, str]:
    return {
        name: value
        for name, value in zip(
            FIELD_ORDER_REGISTRY.get(block.kind, ()), block.fields[1:], strict=False
        )
        if value
    }


def _display_name(obj: IDFBaseModel) -> str | None:
    return next(
        (
            value
            for field_name in (
                "name",
                "zone_name",
                "zone_or_zonelist_name",
                "surface_name",
            )
            if isinstance(value := getattr(obj, field_name, None), str) and value
        ),
        None,
    )


class ParsedIDF:
    """Retain source locations and strictly parsed objects from a single snapshot."""

    def __init__(self, items: list[Block], source: str) -> None:
        self.idf = IDF()
        self.findings: list[Finding] = []
        self.locations: dict[tuple[str, str], Location] = {}
        self.partial_objects: list[dict[str, Any]] = []
        self._names: dict[tuple[str, str], IDFBaseModel] = {}
        self._providers: dict[str, dict[str, list[ProviderDeclaration]]] = {}
        for block in items:
            obj = self._parse(block, source)
            self._register_declarations(block, obj)

    def _parse(self, block: Block, source: str) -> IDFBaseModel | None:
        if not block.terminated or not block.kind:
            self._record_invalid(
                block,
                "invalid_object_syntax",
                "An IDF object needs a type and a terminating semicolon.",
            )
            return None
        try:
            IDF._process_block(
                self.idf, block.fields, strict=True, location=f"{source}:{block.line}"
            )
        except ValidationError as error:
            for entry in error.errors(include_url=False):
                self._record_invalid(
                    block,
                    "invalid_field",
                    entry["msg"],
                    field_name=".".join(map(str, entry["loc"])),
                    keep_partial=False,
                )
            self._record_partial(block)
        except ValueError as error:
            code = (
                "invalid_object"
                if block.kind in FIELD_ORDER_REGISTRY
                else "unrecognized_object_type"
            )
            self._record_invalid(block, code, str(error))
        else:
            objects = self.idf.all_of_type(block.kind)
            key = next(reversed(objects))
            obj = objects[key]
            name = _display_name(obj)
            self.locations[block.kind, key] = Location(block.kind, name, block.line)
            if name is not None:
                self._names.setdefault((block.kind, name.casefold()), obj)
            return obj
        return None

    def _register_declarations(self, block: Block, obj: IDFBaseModel | None) -> None:
        provider_fields = REF_PROVIDERS.get(block.kind)
        if not provider_fields:
            return
        raw = _raw_fields(block)
        for field_name, groups in provider_fields:
            if name := raw.get(field_name):
                declaration = ProviderDeclaration(
                    Location(block.kind, name, block.line, field_name), obj
                )
                for group in groups:
                    index = self._providers.setdefault(group, {})
                    index.setdefault(name.upper(), []).append(declaration)

    def provider_declarations(self, group: str) -> Iterable[list[ProviderDeclaration]]:
        return self._providers.get(group, {}).values()

    def providers(self, name: str, groups: Iterable[str]) -> list[ProviderDeclaration]:
        matches = {}
        for group in groups:
            for declaration in self._providers.get(group, {}).get(name.upper(), ()):
                matches[id(declaration)] = declaration
        return list(matches.values())

    def _record_invalid(
        self,
        block: Block,
        code: str,
        reason: str,
        *,
        field_name: str | None = None,
        keep_partial: bool = True,
    ) -> None:
        self.findings.append(
            Finding(
                "input_error",
                code,
                self._raw_location(block, field_name),
                reason,
                "Correct the indicated object or field using the EnergyPlus 26.1 schema.",
            )
        )
        if keep_partial:
            self._record_partial(block)

    def _raw_location(self, block: Block, field_name: str | None = None) -> Location:
        raw = _raw_fields(block)
        name = (
            raw.get("name")
            or raw.get("zone_name")
            or raw.get("zone_or_zonelist_name")
            or raw.get("surface_name")
        )
        return Location(block.kind or None, name, block.line, field_name)

    def _record_partial(self, block: Block) -> None:
        location = self._raw_location(block)
        self.partial_objects.append(
            {
                "location": asdict(location),
                "raw_field_values": _raw_fields(block),
                "raw_values": block.fields[1:],
                "status": "unvalidated",
            }
        )

    def location(self, obj: IDFBaseModel, field_name: str | None = None) -> Location:
        kind = obj.idf_object_type()
        location = self.locations[kind, obj._idf_obj_key]
        return Location(kind, location.object_name, location.line, field_name)

    def get(self, kind: str, name: str | None) -> IDFBaseModel | None:
        if name:
            return self._names.get(
                (_OBJECT_TYPES.get(kind.casefold(), kind), name.casefold())
            )
        return None

    def reference_findings(self) -> Iterator[Finding]:
        for error in self.idf.validate():
            yield self._reference_finding(error)

    def _reference_finding(self, error: RefError) -> Finding:
        obj = self.idf.all_of_type(error.object_type)[error.object_name]
        issue = describe_reference_issue(error, obj)
        deferred = any(
            provider.target is None
            for provider in self.providers(
                error.referenced_name, error.ref_group.split(", ")
            )
        )
        return Finding(
            "coverage_note" if deferred else "input_error",
            "reference_target_unvalidated"
            if deferred
            else f"reference_{error.error_type}",
            self.location(obj, issue.field),
            (
                f"The reference to {error.referenced_name!r} could not be verified "
                "while a potential provider is unparsed."
                if deferred
                else issue.message
            ),
            (
                "Correct the rejected potential provider, then rerun the reference check."
                if deferred
                else "Correct the referenced name or supply an object of the required type."
            ),
            {
                "referenced_name": error.referenced_name,
                "reference_group": error.ref_group,
            },
        )
