"""Read current package source and its annotations from Oracle."""

import re
import json
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Dict, List

from cancellation import CancellationToken
from databases import get_db_connection


IDENTIFIER = re.compile(r"^[A-Z][A-Z0-9_$#]*$")
QUALIFIED_IDENTIFIER = re.compile(
    r"^(?:[A-Z][A-Z0-9_$#]*\.)?[A-Z][A-Z0-9_$#]*$"
)
SEED_FILE = Path(__file__).parent / "metadata" / "business_concepts.json"
PROCEDURE_PATTERN = re.compile(r"^\s*PROCEDURE\s+([A-Z][A-Z0-9_$#]*)", re.IGNORECASE)
ANNOTATION_PATTERN = re.compile(r"--\s*@([a-z_]+)\s*:\s*(.*?)\s*$", re.IGNORECASE)
SQL_COMMENT_PATTERN = re.compile(r"--[^\r\n]*|/\*[\s\S]*?\*/")


def _normalize_prompt_word(word: str) -> str:
    """Normalize regular English plurals so labels match natural-language requests."""
    value = word.lower()
    if len(value) > 4 and value.endswith("ies"):
        return value[:-3] + "y"
    if len(value) > 3 and value.endswith("s") and not value.endswith(("ss", "us", "is")):
        return value[:-1]
    return value


@dataclass
class PackageMetadata:
    """Current Oracle source and dependencies for one package."""

    owner: str
    package: str
    source: Dict[str, str]
    dependencies: List[Dict[str, str]]
    annotations: List[Dict[str, str]]
    business_concepts: List[Dict[str, Any]] = dataclass_field(default_factory=list)
    direct_concepts: List[Dict[str, Any]] = dataclass_field(default_factory=list)
    lookup_concepts: List[Dict[str, Any]] = dataclass_field(default_factory=list)

    def _procedure_output_fields(self, procedure: str) -> set[str]:
        try:
            query = self.cursor_query(procedure)
        except LookupError:
            return set()

        query = SQL_COMMENT_PATTERN.sub("", query)
        select_match = re.match(r"\s*SELECT\b", query, re.IGNORECASE)
        if not select_match:
            return set()

        select_start = select_match.end()
        depth = 0
        quote = None
        from_start = None
        index = select_start
        while index < len(query):
            char = query[index]
            if quote:
                if char == quote:
                    if index + 1 < len(query) and query[index + 1] == quote:
                        index += 2
                        continue
                    quote = None
            elif char in {"'", '"'}:
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif depth == 0 and query[index:index + 4].upper() == "FROM":
                before = query[index - 1] if index else " "
                after = query[index + 4] if index + 4 < len(query) else " "
                if not (before.isalnum() or before in "_$#") and not (
                    after.isalnum() or after in "_$#"
                ):
                    from_start = index
                    break
            index += 1

        if from_start is None:
            return set()

        expressions = []
        expression_start = 0
        depth = 0
        quote = None
        select_list = query[select_start:from_start]
        index = 0
        while index < len(select_list):
            char = select_list[index]
            if quote:
                if char == quote:
                    if index + 1 < len(select_list) and select_list[index + 1] == quote:
                        index += 2
                        continue
                    quote = None
            elif char in {"'", '"'}:
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "," and depth == 0:
                expressions.append(select_list[expression_start:index].strip())
                expression_start = index + 1
            index += 1
        expressions.append(select_list[expression_start:].strip())

        output_fields = set()
        for expression in expressions:
            if not expression:
                continue
            alias_match = re.search(
                r'(?:\bAS\s+)?(?:"([^"]+)"|([A-Z][A-Z0-9_$#]*))\s*$',
                expression,
                re.IGNORECASE,
            )
            if not alias_match:
                continue
            candidate = (alias_match.group(1) or alias_match.group(2)).upper()
            prefix = expression[:alias_match.start()].strip()
            if not prefix:
                output_fields.add(candidate)
            elif not re.search(r"\bAS\s*$", prefix, re.IGNORECASE):
                output_fields.add(candidate)
        return output_fields

    def field_catalog(self) -> Dict[str, Dict[str, object]]:
        """Build a field catalogue from annotated columns returned by each cursor."""
        body = self.source.get("PACKAGE BODY", "")
        catalog = {}
        output_fields = {}
        for item in self.annotations:
            if not item.get("field") or item["kind"] not in {"label", "concept"}:
                continue

            name = item["field"]
            procedure = item["procedure"]
            if procedure not in output_fields:
                output_fields[procedure] = self._procedure_output_fields(procedure)
            if name.upper() not in output_fields[procedure]:
                continue
            field = catalog.setdefault(
                name,
                {
                    "name": name,
                    "labels": set(),
                    "procedure": procedure,
                    "lineage": [],
                    "business_concepts": [],
                },
            )
            field["labels"].update(
                label.strip() for label in item["value"].split("|") if label.strip()
            )
            expression_pattern = re.compile(
                rf"\b([A-Z][A-Z0-9_$#]*(?:\.[A-Z][A-Z0-9_$#]*)?)"
                rf"\s+(?:AS\s+)?{re.escape(name)}\b",
                re.IGNORECASE,
            )
            for source_line in body.splitlines():
                code = source_line.split("--", 1)[0]
                expression_match = expression_pattern.search(code)
                if expression_match:
                    expression = expression_match.group(1)
                    if (
                        expression.upper() not in {name, "SELECT", "BY", "END", "WHEN"}
                        and expression not in field["lineage"]
                    ):
                        field["lineage"].append(expression)

        for concept in self.business_concepts:
            field_name = concept.get("field", "").upper()
            field = catalog.get(field_name)
            if field is not None:
                field["business_concepts"].append(concept)
        return catalog

    def prompt_context(self, user_prompt: str) -> str:
        """Give Ollama seeded business concepts and current annotated fields."""
        requests_columns = bool(
            re.search(r"\bcolumns?\b", user_prompt, re.IGNORECASE)
        )
        prompt_words = {
            _normalize_prompt_word(word)
            for word in re.findall(r"[A-Z0-9_$#]+", user_prompt, re.IGNORECASE)
            if len(word) > 2
        }
        relevant_annotations = [
            item
            for item in self.annotations
            if item["kind"] in {"business", "capability"}
        ]
        field_lines = []
        fields = self.field_catalog()
        for item in self.annotations:
            value_words = {
                word.lower()
                for word in re.findall(r"[A-Z0-9_$#]+", item["value"], re.IGNORECASE)
                if len(word) > 2
            }
            field = fields.get(item.get("field"))
            field_words = set()
            if field:
                field_words.update(
                    _normalize_prompt_word(word)
                    for word in re.findall(
                        r"[A-Z0-9_$#]+",
                        " ".join(field["labels"]),
                        re.IGNORECASE,
                    )
                )
                field_words.update(
                    _normalize_prompt_word(word)
                    for word in re.findall(
                        r"[A-Z0-9_$#]+",
                        " ".join(field["lineage"]),
                        re.IGNORECASE,
                    )
                )
                field_words.update(
                    _normalize_prompt_word(word)
                    for word in re.findall(r"[A-Z0-9_$#]+", str(field["name"]), re.IGNORECASE)
                )
            if field and prompt_words & field_words and item not in relevant_annotations:
                relevant_annotations.append(item)

        business_lines = [
            f"{item['procedure']}:{item['line']} "
            f"{'-> ' + item['field'] + ' ' if item.get('field') else ''}"
            f"@{item['kind']}: {item['value']}"
            for item in relevant_annotations
        ]
        concept_lines = [
            (
                f"{item['id']} -> {item['field']} "
                f"({item['role']}"
                f"{', ' + item['aggregation'] if item.get('aggregation') else ''}): "
                + " | ".join(item["aliases"])
            )
            for item in self.business_concepts
        ]
        direct_concept_lines = [
            (
                f"{item['id']} -> {item['aggregation']}({item['table']}."
                f"{item['field']}): {item['heading']}"
            )
            for item in self.direct_concepts
        ]
        lookup_concept_lines = [
            (
                f"{item['id']}: {', '.join(item['aliases'])}; "
                f"input={', '.join(item['input']['aliases'])}; "
                f"returns {item['output']['heading']}"
            )
            for item in self.lookup_concepts
        ]
        for name, field in sorted(fields.items()):
            labels = " | ".join(sorted(field["labels"]))
            lineage = f"; source expression: {', '.join(field['lineage'])}" if field["lineage"] else ""
            if requests_columns or prompt_words & {
                _normalize_prompt_word(word)
                for word in re.findall(
                    r"[A-Z0-9_$#]+",
                    f"{name} {labels} {' '.join(field['lineage'])}",
                    re.IGNORECASE,
                )
            }:
                field_lines.append(
                    f"{field['procedure']} {name}: {labels}{lineage}"
                )
        procedure_names = sorted(
            {
                match.group(1).upper()
                for match in re.finditer(
                    r"^\s*PROCEDURE\s+([A-Z][A-Z0-9_$#]*)",
                    self.source.get("PACKAGE BODY", ""),
                    re.IGNORECASE | re.MULTILINE,
                )
            }
        )

        return (
            f"PACKAGE: {self.owner}.{self.package}\n"
            f"PROCEDURES: {', '.join(procedure_names)}\n"
            f"SEEDED BUSINESS CONCEPTS (authoritative when labels overlap):\n"
            + "\n".join(concept_lines)
            + "\n"
            f"ADDITIONAL APPROVED OPERATIONS:\n"
            + "\n".join(direct_concept_lines + lookup_concept_lines)
            + "\n"
            f"BUSINESS MARKERS:\n" + "\n".join(business_lines) + "\n"
            f"APPROVED FIELD LABELS:\n" + "\n".join(field_lines)
        )

    def describe_capabilities(self) -> str:
        """Summarize supported operations from current annotations and seed metadata."""
        def display_alias(aliases: List[str]) -> str:
            return next(
                (alias for alias in aliases if len(alias.split()) > 1),
                aliases[0],
            )

        measures = sorted(
            {
                display_alias(concept["aliases"])
                for concept in self.business_concepts
                if concept["role"] == "measure"
            }
        )
        dimensions = sorted(
            {
                display_alias(concept["aliases"])
                for concept in self.business_concepts
                if concept["role"] == "dimension"
            }
        )
        additional_operations = sorted(
            {
                concept["heading"].lower()
                for concept in self.direct_concepts
            }
            | {
                f"{concept['output']['heading'].lower()} by "
                f"{concept['input']['aliases'][0]}"
                for concept in self.lookup_concepts
            }
        )
        sections = [
            "Use this app to ask read-only questions about the business data "
            "available in this package."
        ]
        if measures:
            sections.append(f"Supported measures: {', '.join(measures)}.")
        if dimensions:
            sections.append(f"Available groupings: {', '.join(dimensions)}.")
        if additional_operations:
            sections.append(
                f"Additional operations: {', '.join(additional_operations)}."
            )
        sections.append(
            "Ask for a specific value, count, lookup, or grouped summary to retrieve results."
        )
        return " ".join(sections)

    def cursor_query(self, procedure_name: str) -> str:
        """Extract a procedure's ref-cursor SELECT and convert inputs to bind names."""
        name = _validate_name(procedure_name, "procedure name")
        body = self.source.get("PACKAGE BODY", "")
        procedure_pattern = re.compile(
            rf"\bPROCEDURE\s+{re.escape(name)}\s*\((.*?)\)\s*(?:IS|AS)\b(.*?)"
            rf"\bEND\s+{re.escape(name)}\s*;",
            re.IGNORECASE | re.DOTALL,
        )
        match = procedure_pattern.search(body)
        if not match:
            raise LookupError(f"Procedure {name} was not found in {self.package}.")

        signature, procedure_body = match.groups()
        query_match = re.search(
            r"\bOPEN\s+[A-Z][A-Z0-9_$#]*\s+FOR\s+(SELECT\b[\s\S]*?)\s*;",
            procedure_body,
            re.IGNORECASE,
        )
        if not query_match:
            raise LookupError(f"Procedure {name} does not open a SELECT ref cursor.")

        query = query_match.group(1).strip()
        input_parameters = re.findall(
            r"^\s*([A-Z][A-Z0-9_$#]*)\s+IN(?:\s|$)",
            signature,
            re.IGNORECASE | re.MULTILINE,
        )
        for parameter in input_parameters:
            query = re.sub(
                rf"(?<![A-Z0-9_$#]){re.escape(parameter)}(?![A-Z0-9_$#])",
                f":{parameter.upper()}",
                query,
                flags=re.IGNORECASE,
            )
        return query


def _validate_name(name: str, label: str) -> str:
    value = name.strip().upper()
    if not IDENTIFIER.fullmatch(value):
        raise ValueError(f"Invalid Oracle {label}: {name!r}")
    return value


def _read_annotations(source: str) -> List[Dict[str, str]]:
    annotations = []
    procedure = ""

    for line_number, line in enumerate(source.splitlines(), start=1):
        procedure_match = PROCEDURE_PATTERN.match(line)
        if procedure_match:
            procedure = procedure_match.group(1).upper()

        annotation_match = ANNOTATION_PATTERN.search(line)
        if annotation_match:
            item = {
                "procedure": procedure,
                "line": str(line_number),
                "kind": annotation_match.group(1).lower(),
                "value": annotation_match.group(2).strip(),
            }
            code = line[:annotation_match.start()].strip()
            field_match = re.search(r"\b([A-Z][A-Z0-9_$#]*)\s*,?\s*$", code, re.IGNORECASE)
            if field_match:
                item["field"] = field_match.group(1).upper()
            annotations.append(item)

    return annotations


def _read_business_concepts(owner: str, package: str) -> List[Dict[str, Any]]:
    try:
        with SEED_FILE.open(encoding="utf-8") as seed_file:
            document = json.load(seed_file)
    except FileNotFoundError as error:
        raise LookupError(f"Business concept seed file is missing: {SEED_FILE}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Business concept seed file is invalid JSON: {SEED_FILE}") from error

    package_key = f"{owner}.{package}".upper()
    package_data = document.get("packages", {}).get(package_key, {})
    procedure_data = package_data.get("procedures", {})
    concepts = []
    for procedure, procedure_config in procedure_data.items():
        for item in procedure_config.get("concepts", []):
            concept = {
                **item,
                "procedure": procedure.upper(),
                "field": str(item.get("field", "")).upper(),
            }
            if (
                not IDENTIFIER.fullmatch(concept["field"])
                or not isinstance(concept.get("id"), str)
                or not isinstance(concept.get("aliases"), list)
                or not concept["aliases"]
                or not all(isinstance(alias, str) and alias.strip() for alias in concept["aliases"])
                or concept.get("role") not in {"measure", "dimension"}
            ):
                raise ValueError(
                    f"Invalid seeded business concept in {package_key}.{procedure}."
                )
            aggregation = concept.get("aggregation")
            if aggregation is not None and aggregation not in {
                "SUM",
                "COUNT",
                "COUNT_DISTINCT",
                "AVG",
                "MIN",
                "MAX",
            }:
                raise ValueError(f"Invalid aggregation for seeded concept {concept['id']}.")
            if concept["role"] == "measure" and aggregation is None:
                raise ValueError(f"Seeded measure {concept['id']} requires an aggregation.")
            if concept["role"] == "dimension" and aggregation is not None:
                raise ValueError(f"Seeded dimension {concept['id']} cannot have an aggregation.")
            concepts.append(concept)
    return concepts


def _read_direct_concepts(owner: str, package: str) -> List[Dict[str, Any]]:
    try:
        with SEED_FILE.open(encoding="utf-8") as seed_file:
            document = json.load(seed_file)
    except FileNotFoundError as error:
        raise LookupError(f"Business concept seed file is missing: {SEED_FILE}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Business concept seed file is invalid JSON: {SEED_FILE}") from error

    package_key = f"{owner}.{package}".upper()
    package_data = document.get("packages", {}).get(package_key, {})
    concepts = []
    for item in package_data.get("direct_concepts", []):
        concept = {
            **item,
            "table": str(item.get("table", "")).upper(),
            "field": str(item.get("field", "")).upper(),
            "aggregation": str(item.get("aggregation", "")).upper(),
        }
        if (
            not IDENTIFIER.fullmatch(concept["table"])
            or not IDENTIFIER.fullmatch(concept["field"])
            or concept["aggregation"] not in {"COUNT"}
            or not isinstance(concept.get("id"), str)
            or not isinstance(concept.get("heading"), str)
            or not concept["heading"].strip()
            or not isinstance(concept.get("aliases"), list)
            or not concept["aliases"]
            or not all(
                isinstance(alias, str) and alias.strip()
                for alias in concept["aliases"]
            )
        ):
            raise ValueError(f"Invalid direct business concept in {package_key}.")
        concepts.append(concept)
    return concepts


def _read_lookup_concepts(owner: str, package: str) -> List[Dict[str, Any]]:
    try:
        with SEED_FILE.open(encoding="utf-8") as seed_file:
            document = json.load(seed_file)
    except FileNotFoundError as error:
        raise LookupError(f"Business concept seed file is missing: {SEED_FILE}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Business concept seed file is invalid JSON: {SEED_FILE}") from error

    package_key = f"{owner}.{package}".upper()
    package_data = document.get("packages", {}).get(package_key, {})
    concepts = []
    for item in package_data.get("lookup_concepts", []):
        concept = dict(item)
        source = concept.get("from", {})
        input_config = concept.get("input", {})
        output = concept.get("output", {})
        joins = concept.get("joins", [])
        conditions = concept.get("conditions", [])
        aliases = concept.get("aliases", [])
        input_aliases = input_config.get("aliases", [])
        if not isinstance(joins, list) or not isinstance(conditions, list):
            raise ValueError(f"Invalid lookup business concept in {package_key}.")
        simple_identifiers = [
            source.get("table"),
            source.get("alias"),
            input_config.get("bind"),
            *(join.get("table") for join in joins if isinstance(join, dict)),
            *(join.get("alias") for join in joins if isinstance(join, dict)),
            *(condition.get("bind") for condition in conditions if isinstance(condition, dict)),
        ]
        qualified_identifiers = [
            input_config.get("field"),
            output.get("field"),
            *(
                identifier
                for join in joins
                if isinstance(join, dict)
                for identifier in (join.get("left"), join.get("right"))
            ),
            *(
                condition.get("field")
                for condition in conditions
                if isinstance(condition, dict)
            ),
        ]
        if (
            not isinstance(concept.get("id"), str)
            or not all(
                isinstance(value, str) and IDENTIFIER.fullmatch(value)
                for value in simple_identifiers
            )
            or not all(
                isinstance(value, str) and QUALIFIED_IDENTIFIER.fullmatch(value)
                for value in qualified_identifiers
            )
            or not isinstance(input_config.get("pattern"), str)
            or not input_config["pattern"]
            or not isinstance(output.get("heading"), str)
            or not output["heading"].strip()
            or not aliases
            or not all(isinstance(alias, str) and alias.strip() for alias in aliases)
            or not input_aliases
            or not all(
                isinstance(alias, str) and alias.strip()
                for alias in input_aliases
            )
            or any(join.get("type") != "JOIN" for join in joins)
            or any(
                condition.get("operator") != "="
                for condition in conditions
            )
        ):
            raise ValueError(f"Invalid lookup business concept in {package_key}.")
        concept["from"] = {**source, "table": source["table"].upper(), "alias": source["alias"].upper()}
        concept["input"] = {
            **input_config,
            "field": input_config["field"].upper(),
            "bind": input_config["bind"].upper(),
        }
        concept["output"] = {**output, "field": output["field"].upper()}
        concept["joins"] = [
            {
                **join,
                "table": join["table"].upper(),
                "alias": join["alias"].upper(),
                "left": join["left"].upper(),
                "right": join["right"].upper(),
            }
            for join in joins
        ]
        concept["conditions"] = [
            {
                **condition,
                "field": condition["field"].upper(),
                "bind": condition["bind"].upper(),
            }
            for condition in conditions
        ]
        concepts.append(concept)
    return concepts


def read_package_metadata(
    package: str,
    owner: str = "PQTEPF",
    database_name: str = "PQTEPF",
    cancellation: CancellationToken | None = None,
) -> PackageMetadata:
    """Load fresh package source, dependencies, and inline annotations."""
    package_name = _validate_name(package, "package name")
    owner_name = _validate_name(owner, "owner")

    connection = get_db_connection(database_name)
    if cancellation is not None:
        cancellation.attach_cancel_callback(connection.cancel)
    try:
        if cancellation is not None:
            cancellation.check()
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT type, line, text
                FROM all_source
                WHERE owner = :owner
                  AND name = :package
                  AND type IN ('PACKAGE', 'PACKAGE BODY')
                ORDER BY type, line
                """,
                owner=owner_name,
                package=package_name,
            )
            source_rows = cursor.fetchall()
            if cancellation is not None:
                cancellation.check()
            cursor.execute(
                """
                SELECT DISTINCT referenced_owner, referenced_name,
                                referenced_type, referenced_link_name
                FROM all_dependencies
                WHERE owner = :owner
                  AND name = :package
                  AND type IN ('PACKAGE', 'PACKAGE BODY')
                  AND referenced_name != :package
                ORDER BY referenced_owner, referenced_name
                """,
                owner=owner_name,
                package=package_name,
            )
            dependency_rows = cursor.fetchall()
    finally:
        if cancellation is not None:
            cancellation.clear_cancel_callback()
        connection.close()

    if not source_rows:
        raise LookupError(f"Oracle package {owner_name}.{package_name} was not found.")

    source: Dict[str, str] = {}
    for object_type, _, text in source_rows:
        source[object_type] = source.get(object_type, "") + text

    body = source.get("PACKAGE BODY", "")
    dependencies = [
        {
            "owner": row[0] or "",
            "name": row[1],
            "type": row[2],
            "link": row[3] or "",
        }
        for row in dependency_rows
    ]

    metadata = PackageMetadata(
        owner=owner_name,
        package=package_name,
        source=source,
        dependencies=dependencies,
        annotations=_read_annotations(body),
        business_concepts=_read_business_concepts(owner_name, package_name),
        direct_concepts=_read_direct_concepts(owner_name, package_name),
        lookup_concepts=_read_lookup_concepts(owner_name, package_name),
    )
    fields = metadata.field_catalog()
    for concept in metadata.business_concepts:
        if concept["field"] not in fields:
            raise ValueError(
                "Seeded business concept "
                f"{concept['id']} maps to {concept['field']}, which is not returned "
                f"by {concept['procedure']}."
            )
        if fields[concept["field"]]["procedure"] != concept["procedure"]:
            raise ValueError(
                "Seeded business concept "
                f"{concept['id']} is assigned to the wrong procedure."
            )
    return metadata
