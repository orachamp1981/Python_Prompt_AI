"""Interpret a business prompt and compose SQL from live package logic."""

import calendar
import json
import os
import re
from datetime import date
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from cancellation import CancellationToken
from package_metadata import PackageMetadata


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:latest"
DEFAULT_INTENT_MODEL = "qwen2.5:1.5b"
AGGREGATIONS = {"SUM", "COUNT", "AVG", "MIN", "MAX", "COUNT_DISTINCT"}
IDENTIFIER_PATTERN = re.compile(r"^[A-Z][A-Z0-9_$#]*$")
CASH_VALUE_REQUEST_PATTERN = re.compile(r"\bcash[\s-]+values?\b", re.IGNORECASE)
DOCUMENT_CODE_PATTERN = re.compile(r"\b[A-Z]{2,}\d{6,}\b", re.IGNORECASE)
INTENT_RESPONSE_FORMAT = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["information", "business_query"],
        }
    },
    "required": ["intent"],
}

RESPONSE_FORMAT = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["ready", "clarify", "unsupported", "informational"],
        },
        "measure_field": {"type": ["string", "null"]},
        "aggregation": {"type": ["string", "null"]},
        "group_by": {"type": "array", "items": {"type": "string"}},
        "select_columns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "aggregation": {
                        "type": ["string", "null"],
                        "enum": ["SUM", "COUNT", "AVG", "MIN", "MAX", "COUNT_DISTINCT", None],
                    },
                    "alias": {"type": "string"},
                    "date_grain": {
                        "type": ["string", "null"],
                        "enum": ["YEAR", "MONTH", "DAY", None],
                    },
                },
                "required": ["field", "aggregation", "alias", "date_grain"],
            },
        },
        "message": {"type": "string"},
    },
    "required": [
        "status",
        "measure_field",
        "aggregation",
        "group_by",
        "select_columns",
        "message",
    ],
}

SYSTEM_PROMPT = """You understand natural-language requests about the application and
the business data represented by its package metadata.
Return only the required JSON object. Do not write SQL.
When the user is asking what the application or available data can provide, how it
works, or what kind of result a prompt can produce, return status "informational".
Answer the actual question in message using only PACKAGE METADATA, including its
business annotations, seeded concepts, and approved fields. Do not assume the user
wants a data query just because the question mentions a prompt or business data.
Do not ask for a measure for an informational question. For an informational
response, use null measure_field and aggregation, empty group_by and
select_columns, and put the complete answer in message.
For a request to retrieve or calculate business data, use a query-planning status.
Use only canonical field names in APPROVED FIELD LABELS. For a phrase such as
"region-wise", choose the uniquely matching Region field in group_by.
Choose the requested numeric measure and its operation. If the request does not
identify a measure unambiguously, ask for clarification. Do not invent fields,
values, or filters. Package scope is established by its @business and
@capability annotations and their approved fields; users do not need to repeat
the annotation value verbatim. Use status "clarify" whenever the measure,
aggregation, or scope needs user input; do not use "unsupported" to ask a
question. Return unsupported only when the requested subject is outside the
annotated package scope.
For a question asking how many values a categorical field has, use
COUNT_DISTINCT on that annotated field and no group_by fields. Do not ask for a
numeric measure when the question is asking for the number of categories.
Use the package field labels and metadata to identify the requested field.
When the user requests named output columns, use select_columns to return every
requested column in order. Each field must be selected from APPROVED FIELD
LABELS. Use aggregation null for a grouped dimension, and set date_grain to
MONTH, YEAR, or DAY only when the requested output groups a date field at that
grain. Use COUNT for a requested case/record count and the metadata-matched
field. Use the user's requested output heading as alias. Set select_columns to
an empty array when no explicit multi-column output is requested. When
select_columns is present, it is the complete requested result: do not add
unrequested group_by fields. Each date column has aggregation null; each
aggregated column has date_grain null. A clarification is not needed when all
requested columns map unambiguously to annotated fields and aggregations.
Resolve every requested column independently. A date output is a grouping
dimension, never an aggregated value; a case/record total counts an annotated
record identifier rather than a status or amount. Resolve measure labels using
all matching annotation labels and request context, preferring the field whose
annotations explain the requested measure rather than one shared word.
"""


def _is_information_request(
    prompt: str,
    model: Optional[str] = None,
    ollama_url: Optional[str] = None,
    cancellation: Optional[CancellationToken] = None,
) -> bool:
    """Use a small local model to distinguish questions from record retrieval."""
    base_url = (
        ollama_url or os.environ.get("OLLAMA_URL") or DEFAULT_OLLAMA_URL
    ).rstrip("/")
    model_name = (
        model
        or os.environ.get("PROMPT_INTENT_MODEL")
        or DEFAULT_INTENT_MODEL
    )
    request_body = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Classify the user's intent, not particular words. Use "
                    "\"information\" when they are asking to understand the "
                    "application, its purpose, supported capabilities, or what "
                    "kind of information is available. Use \"business_query\" "
                    "only when they want actual business records retrieved, "
                    "counted, calculated, filtered, or grouped. A question "
                    "about the tool is not a request to retrieve records."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "format": INTENT_RESPONSE_FORMAT,
        "stream": False,
        "keep_alive": "10m",
        "options": {"temperature": 0, "num_predict": 16},
    }
    request = Request(
        f"{base_url}/api/chat",
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        payload = _read_ollama_json(request, cancellation, timeout=20)
    except HTTPError as error:
        if cancellation is not None:
            cancellation.check()
        raise RuntimeError(
            f"Local intent model returned HTTP {error.code}."
        ) from error
    except (URLError, TimeoutError) as error:
        if cancellation is not None:
            cancellation.check()
        raise RuntimeError(
            f"Local intent model did not respond at {base_url}; "
            "check Ollama and the intent model."
        ) from error

    content = payload.get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("Local intent model did not return classification content.")
    try:
        classification = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("Local intent model returned invalid classification JSON.") from error
    if not isinstance(classification, dict) or classification.get("intent") not in {
        "information",
        "business_query",
    }:
        raise ValueError("Local intent model returned an unsupported classification.")
    return classification["intent"] == "information"


def _read_ollama_json(
    request: Request,
    cancellation: Optional[CancellationToken],
    timeout: int,
) -> Dict[str, Any]:
    """Read a complete Ollama chat response, closing its stream on cancellation."""
    if cancellation is None:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    cancellation.check()
    body = json.loads(request.data.decode("utf-8"))
    body["stream"] = True
    request = Request(
        request.full_url,
        data=json.dumps(body).encode("utf-8"),
        headers=dict(request.header_items()),
        method="POST",
    )
    chunks = []
    try:
        with urlopen(request, timeout=timeout) as response:
            cancellation.attach_cancel_callback(response.close)
            for line in response:
                cancellation.check()
                if not line.strip():
                    continue
                payload = json.loads(line.decode("utf-8"))
                if payload.get("error"):
                    raise RuntimeError(f"Local Ollama error: {payload['error']}")
                content = payload.get("message", {}).get("content")
                if isinstance(content, str):
                    chunks.append(content)
            cancellation.check()
    except (OSError, ValueError):
        cancellation.check()
        raise
    finally:
        cancellation.clear_cancel_callback()
    return {"message": {"content": "".join(chunks)}}


def _parse_request_date_range(prompt: str) -> Optional[Dict[str, str]]:
    """Resolve a named calendar month to inclusive Oracle DATE bind values."""
    month_lookup = {}
    for month in range(1, 13):
        month_lookup[calendar.month_name[month].lower()] = month
        month_lookup[calendar.month_abbr[month].lower()] = month
    names = "|".join(sorted(month_lookup, key=len, reverse=True))
    match = re.search(rf"\b({names})\s+((?:19|20)\d{{2}})\b", prompt, re.IGNORECASE)
    if not match:
        return None

    month = month_lookup[match.group(1).lower()]
    year = int(match.group(2))
    last_day = calendar.monthrange(year, month)[1]
    return {
        "P_SDATE": date(year, month, 1).isoformat(),
        "P_EDATE": date(year, month, last_day).isoformat() + " 23:59:59",
    }


def _cash_value_lookup(prompt: str, metadata: PackageMetadata) -> Optional[Dict[str, Any]]:
    if not CASH_VALUE_REQUEST_PATTERN.search(prompt):
        return None

    capability = next(
        (
            item
            for item in metadata.annotations
            if item["kind"] == "capability"
            and "cash" in item["value"].lower()
            and "value" in item["value"].lower()
        ),
        None,
    )
    if capability is None:
        return None

    document_codes = DOCUMENT_CODE_PATTERN.findall(prompt)
    if len(document_codes) != 1:
        return {
            "status": "clarify",
            "message": (
                "Please provide exactly one policy or document code for the cash-value lookup."
            ),
            "operation": "cash_value_lookup",
            "sql": None,
            "parameters": {},
            "required_parameters": [],
        }

    return {
        "status": "ready",
        "operation": "cash_value_lookup",
        "procedure": capability["procedure"],
        "owner": metadata.owner,
        "package": metadata.package,
        "document_code": document_codes[0].upper(),
        "measure_field": "CASH_VALUE",
        "aggregation": "LOOKUP",
        "group_by": [],
        "user_prompt": prompt,
        "parameters": {},
        "required_parameters": [],
    }


def _read_plan(content: str, metadata: PackageMetadata) -> Dict[str, Any]:
    try:
        plan = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("Ollama did not return valid query-plan JSON.") from error
    if not isinstance(plan, dict):
        raise ValueError("Ollama query plan must be a JSON object.")

    status = plan.get("status")
    if status not in {"ready", "clarify", "unsupported", "informational"}:
        raise ValueError("Ollama returned an unsupported query-plan status.")
    if not isinstance(plan.get("message"), str):
        raise ValueError("Ollama query-plan message must be text.")
    if status == "informational":
        plan["measure_field"] = None
        plan["aggregation"] = None
        plan["group_by"] = []
        plan["select_columns"] = []
        return plan
    if status != "ready":
        if status == "unsupported" and "clarif" in plan["message"].lower():
            plan["status"] = "clarify"
        return plan

    fields = metadata.field_catalog()
    measure = plan.get("measure_field")
    groups = plan.get("group_by")
    aggregation = str(plan.get("aggregation", "")).upper()
    if not isinstance(measure, str) or measure.upper() not in fields:
        raise ValueError("Ollama selected a measure that is not an annotated package field.")
    if not isinstance(groups, list) or not all(
        isinstance(field, str) and field.upper() in fields for field in groups
    ):
        raise ValueError("Ollama selected a grouping field that is not annotated.")
    if aggregation not in AGGREGATIONS:
        raise ValueError("Ollama selected an unsupported aggregation.")
    if measure.upper() in {field.upper() for field in groups}:
        raise ValueError("Measure and grouping fields cannot be the same.")

    plan["measure_field"] = measure.upper()
    plan["group_by"] = [field.upper() for field in groups]
    plan["aggregation"] = aggregation
    columns = plan.get("select_columns", [])
    if not isinstance(columns, list):
        raise ValueError("Ollama selected columns must be an array.")
    validated_columns = []
    for column in columns:
        if not isinstance(column, dict):
            raise ValueError("Ollama selected column must be an object.")
        field = column.get("field")
        aggregation = column.get("aggregation")
        alias = column.get("alias")
        date_grain = column.get("date_grain")
        if not isinstance(field, str) or not IDENTIFIER_PATTERN.fullmatch(
            field.upper()
        ):
            raise ValueError("Ollama selected an invalid result field.")
        if aggregation is not None and not isinstance(aggregation, str):
            raise ValueError("Ollama selected an invalid column aggregation.")
        if not isinstance(alias, str) or not re.fullmatch(
            r"[A-Z][A-Z0-9 _-]{0,63}", alias, re.IGNORECASE
        ):
            raise ValueError("Ollama selected an invalid result column heading.")
        if date_grain is not None and str(date_grain).upper() not in {
            "YEAR",
            "MONTH",
            "DAY",
        }:
            raise ValueError("Ollama selected an unsupported date grouping.")
        if date_grain is not None and aggregation is not None:
            raise ValueError("Date grouping columns cannot have an aggregation.")
        validated_columns.append(
            {
                **column,
                "field": field.upper(),
                "aggregation": str(aggregation).upper() if aggregation else None,
                "alias": alias,
                "date_grain": str(date_grain).upper() if date_grain else None,
            }
        )
    plan["select_columns"] = validated_columns
    return plan


def _resolve_annotated_measure(plan: Dict[str, Any], metadata: PackageMetadata) -> bool:
    """Resolve a clarification when the prompt exactly names one annotated measure."""
    fields = metadata.field_catalog()
    measure = plan.get("measure_field")
    groups = plan.get("group_by")
    aggregation = str(plan.get("aggregation", "")).upper()
    if (
        not isinstance(measure, str)
        or measure.upper() not in fields
        or not isinstance(groups, list)
        or not all(isinstance(field, str) and field.upper() in fields for field in groups)
        or aggregation not in AGGREGATIONS
    ):
        return False

    prompt = " " + " ".join(
        re.findall(r"[A-Z0-9_$#]+", plan.get("user_prompt", "").upper())
    ) + " "
    group_names = {field.upper() for field in groups}
    matched_measures = set()
    for name, field in fields.items():
        if name.upper() in group_names:
            continue
        for label in field["labels"]:
            normalized_label = " ".join(
                re.findall(r"[A-Z0-9_$#]+", label.upper())
            )
            if normalized_label and f" {normalized_label} " in prompt:
                matched_measures.add(name.upper())

    if matched_measures != {measure.upper()}:
        return False

    plan["status"] = "ready"
    plan["measure_field"] = measure.upper()
    plan["group_by"] = [field.upper() for field in groups]
    plan["aggregation"] = aggregation
    plan["message"] = "Measure matched to its unique package annotation label."
    return True


def _resolve_distinct_dimension_count(
    plan: Dict[str, Any],
    prompt: str,
    metadata: PackageMetadata,
) -> bool:
    """Resolve count-distinct intents using annotated dimensions, not model confirmation."""
    if plan.get("status") != "clarify":
        return False
    if not re.search(r"\b(how many|number of|count)\b", prompt, re.IGNORECASE):
        return False

    fields = metadata.field_catalog()
    aggregation = str(plan.get("aggregation", "")).upper()
    measure = plan.get("measure_field")
    groups = plan.get("group_by")
    field_name = None
    if (
        aggregation == "COUNT_DISTINCT"
        and isinstance(measure, str)
        and measure.upper() in fields
        and groups == []
    ):
        field_name = measure.upper()
    elif aggregation == "COUNT" and str(measure).upper() == "COUNT_DISTINCT":
        if not isinstance(groups, list) or len(groups) != 1:
            return False
        if not isinstance(groups[0], str) or groups[0].upper() not in fields:
            return False
        field_name = groups[0].upper()
    if field_name is None:
        field_name = _match_annotated_dimension(prompt, metadata)
        if field_name is None:
            return False

    plan.update(
        status="ready",
        measure_field=field_name.upper(),
        aggregation="COUNT_DISTINCT",
        group_by=[],
        message="Count distinct values of the annotated dimension.",
    )
    return True


def _match_annotated_dimension(prompt: str, metadata: PackageMetadata) -> Optional[str]:
    """Match count nouns against annotated labels, requiring one best field match."""
    ignored_words = {
        "how", "many", "number", "count", "distinct", "of", "the", "are",
        "there", "in", "inside", "within", "scope", "system", "all",
    }

    def normalize(word: str) -> str:
        word = word.lower()
        if len(word) > 4 and word.endswith("ies"):
            return word[:-3] + "y"
        if len(word) > 5 and word.endswith("ses"):
            return word[:-2]
        if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
            return word[:-1]
        return word

    requested_terms = {
        normalize(word)
        for word in re.findall(r"[A-Z0-9_$#]+", prompt, re.IGNORECASE)
        if normalize(word) not in ignored_words
    }
    if not requested_terms:
        return None

    field_scores = {}
    for name, field in metadata.field_catalog().items():
        aliases = [
            {
                normalize(word)
                for word in re.findall(r"[A-Z0-9_$#]+", alias, re.IGNORECASE)
                if normalize(word) not in ignored_words
            }
            for alias in field["labels"]
        ]
        aliases.append(
            {
                normalize(word)
                for word in re.findall(r"[A-Z0-9_$#]+", name, re.IGNORECASE)
            }
        )
        scores = [
            len(requested_terms & alias) / len(alias)
            for alias in aliases
            if alias and alias <= requested_terms
        ]
        if scores:
            field_scores[name.upper()] = max(scores)

    if not field_scores:
        return None
    best_score = max(field_scores.values())
    matches = [name for name, score in field_scores.items() if score == best_score]
    return matches[0] if len(matches) == 1 else None


def _distinct_dimension_count_plan(
    prompt: str,
    metadata: PackageMetadata,
) -> Optional[Dict[str, Any]]:
    """Build count-distinct plans directly from a unique live annotation match."""
    if not re.search(r"\b(how many|number of|count)\b", prompt, re.IGNORECASE):
        return None
    field_name = _match_annotated_dimension(prompt, metadata)
    if field_name is None:
        return None
    plan = {
        "status": "ready",
        "measure_field": field_name,
        "aggregation": "COUNT_DISTINCT",
        "group_by": [],
        "message": "Count distinct values of the annotated dimension.",
        "user_prompt": prompt,
    }
    return _build_sql(plan, metadata)


def _distinct_dimension_headings(
    prompt: str,
    field_name: str,
    metadata: PackageMetadata,
) -> tuple[str, str]:
    """Build count and value-list headings from the matching live field label."""
    field = metadata.field_catalog()[field_name]
    prompt_terms = {
        _normalize_count_word(word)
        for word in re.findall(r"[A-Z0-9_$#]+", prompt, re.IGNORECASE)
    }
    matching_labels = []
    for label in field["labels"]:
        label_terms = {
            _normalize_count_word(word)
            for word in re.findall(r"[A-Z0-9_$#]+", label, re.IGNORECASE)
        }
        if label_terms and label_terms <= prompt_terms:
            matching_labels.append((len(label_terms), len(label), label))

    label = (
        max(matching_labels, key=lambda item: (item[0], -item[1]))[2]
        if matching_labels
        else field_name.replace("_", " ").title()
    )
    plural = _pluralize_label(label)
    return f"Total Number of {plural}", plural


def _normalize_count_word(word: str) -> str:
    value = word.lower()
    if len(value) > 4 and value.endswith("ies"):
        return value[:-3] + "y"
    if len(value) > 5 and value.endswith("ses"):
        return value[:-2]
    if len(value) > 3 and value.endswith("s") and not value.endswith(("ss", "us", "is")):
        return value[:-1]
    return value


def _pluralize_label(label: str) -> str:
    words = label.split()
    last = words[-1]
    lowered = last.lower()
    if lowered.endswith("y") and len(lowered) > 1 and lowered[-2] not in "aeiou":
        words[-1] = last[:-1] + "ies"
    elif lowered.endswith(("s", "x", "z", "ch", "sh")):
        words[-1] = last + "es"
    else:
        words[-1] = last + "s"
    return " ".join(words)


def _field_display_label(prompt: str, field: Dict[str, Any]) -> str:
    """Choose a user-facing annotation label instead of exposing a column name."""
    prompt_terms = {
        _normalize_count_word(word)
        for word in re.findall(r"[A-Z0-9_$#]+", prompt, re.IGNORECASE)
    }
    matching_labels = []
    for label in field["labels"]:
        label_terms = {
            _normalize_count_word(word)
            for word in re.findall(r"[A-Z0-9_$#]+", label, re.IGNORECASE)
        }
        if label_terms and label_terms <= prompt_terms:
            matching_labels.append((len(label_terms), len(label), label))

    if matching_labels:
        return max(matching_labels, key=lambda item: (item[0], -item[1]))[2]
    return str(field["name"]).replace("_", " ").title()


def _semantic_terms(value: str) -> set[str]:
    words = {
        _normalize_count_word(word)
        for word in re.findall(r"[A-Z0-9_$#]+", value, re.IGNORECASE)
    }
    if re.search(r"\btop\s+up\b", value, re.IGNORECASE):
        words.add("topup")
    if words & {"register", "registered", "registration"}:
        words.update({"register", "registered", "registration", "submission", "entry"})
    if words & {"submit", "submitted", "submission"}:
        words.update({"submit", "submitted", "submission", "register", "entry"})
    if words & {"record", "recorded", "created"}:
        words.update({"record", "recorded", "created", "entry"})
    return words


def _column_explanation(prompt: str, alias: str) -> str:
    alias_pattern = r"\s+".join(re.escape(word) for word in alias.split())
    match = re.search(
        rf"\b{alias_pattern}\s+means?\s+(.+?)(?:[.;\n]|$)",
        prompt,
        re.IGNORECASE,
    )
    return match.group(1) if match else ""


def _requested_output_aliases(prompt: str) -> list[str]:
    match = re.search(
        r"\bcolumns?\s+(?:should\s+be|should|are|be)\s+(.+?)(?=[.;?!\n]|$)",
        prompt,
        re.IGNORECASE,
    )
    if not match:
        return []
    column_text = match.group(1)
    explanation = re.search(
        r",\s*(?:and\s+)?[^,;.!?]+?\s+means?\b",
        column_text,
        re.IGNORECASE,
    )
    if explanation:
        column_text = column_text[:explanation.start()]
    return [
        re.sub(r"^(?:and|,)\s*", "", item.strip(), flags=re.IGNORECASE)
        for item in re.split(r",", column_text)
        if item.strip().lower() not in {"and"}
    ]


def _is_date_field(name: str, field: Dict[str, Any]) -> bool:
    return name.upper().endswith(("_DATE", "_DT")) or any(
        re.search(r"\b(date|birth|dob)\b", label, re.IGNORECASE)
        for label in field["labels"]
    )


def _match_seeded_concept(
    text: str,
    metadata: PackageMetadata,
    role: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    request_terms = _semantic_terms(text)
    scores = []
    for concept in metadata.business_concepts:
        if role and concept["role"] != role:
            continue
        for alias in concept["aliases"]:
            alias_terms = _semantic_terms(alias)
            if alias_terms and alias_terms <= request_terms:
                scores.append((len(alias_terms), concept))

    if not scores:
        return None
    best_score = max(score for score, _ in scores)
    best_concepts = {
        concept["id"]: concept
        for score, concept in scores
        if score == best_score
    }
    return next(iter(best_concepts.values())) if len(best_concepts) == 1 else None


def _requested_date_group_grain(prompt: str) -> Optional[str]:
    patterns = {
        "YEAR": (
            r"\b(?:year[- ]wise|yearly|annual(?:ly)?)\b",
            r"\b(?:by|per|across|group(?:ed)?\s+by)\b.{0,40}\byears?\b",
            r"\bregistration\s+year\b",
        ),
        "MONTH": (
            r"\b(?:month[- ]wise|monthly)\b",
            r"\b(?:by|per|across|group(?:ed)?\s+by)\b.{0,40}\bmonths?\b",
            r"\bregistration\s+month\b",
        ),
        "DAY": (
            r"\b(?:day[- ]wise|daily)\b",
            r"\b(?:by|per|across|group(?:ed)?\s+by)\b.{0,40}\bdays?\b",
        ),
    }
    for grain, expressions in patterns.items():
        if any(re.search(expression, prompt, re.IGNORECASE) for expression in expressions):
            return grain
    return None


def _resolve_output_columns(
    plan: Dict[str, Any],
    prompt: str,
    metadata: PackageMetadata,
) -> bool:
    """Normalize model column choices against field labels and requested semantics."""
    columns = plan.get("select_columns")
    if not isinstance(columns, list) or not columns:
        return False

    fields = metadata.field_catalog()
    if not fields:
        return False

    requested_aliases = _requested_output_aliases(prompt)
    if requested_aliases:
        model_columns = [column for column in columns if isinstance(column, dict)]
        aligned_columns = []
        for alias in requested_aliases:
            alias_terms = _semantic_terms(alias)
            scored_columns = []
            for model_column in model_columns:
                model_alias = model_column.get("alias")
                if not isinstance(model_alias, str):
                    continue
                model_terms = _semantic_terms(model_alias)
                overlap = len(alias_terms & model_terms)
                if overlap:
                    scored_columns.append(
                        (
                            overlap / len(alias_terms | model_terms),
                            overlap,
                            model_column,
                        )
                    )
            if not scored_columns:
                aligned_columns.append({"alias": alias})
                continue
            best_score = max((item[0], item[1]) for item in scored_columns)
            best = [
                item[2]
                for item in scored_columns
                if (item[0], item[1]) == best_score
            ]
            aligned_columns.append(
                dict(best[0], alias=alias) if len(best) == 1 else {"alias": alias}
            )
        columns = aligned_columns

    resolved = []
    for column in columns:
        if not isinstance(column, dict) or not isinstance(column.get("alias"), str):
            return False
        alias = column["alias"].strip()
        alias_terms = _semantic_terms(alias)
        explanation = _column_explanation(prompt, alias)
        context_terms = _semantic_terms(prompt + " " + explanation + " " + alias)
        alias_lower = alias.lower()
        field_name = column.get("field")
        aggregation = column.get("aggregation")
        date_grain = column.get("date_grain")
        seeded_concept = _match_seeded_concept(
            alias + " " + explanation,
            metadata,
        )
        if (
            seeded_concept is None
            and alias_lower in {"year", "month", "day", "date", "period"}
        ):
            seeded_concept = _match_seeded_concept(
                prompt,
                metadata,
                role="dimension",
            )

        is_case_count = bool(
            re.search(r"\b(case(?:s)?|record(?:s)?|count)\b", alias_lower)
            and (str(aggregation).upper() == "COUNT" or alias_lower != "count")
        )
        if seeded_concept is not None:
            field_name = seeded_concept["field"]
            aggregation = seeded_concept.get("aggregation")
            date_grain = None
            if seeded_concept["role"] == "dimension":
                grain_terms = _semantic_terms(alias + " " + explanation)
                date_grain = (
                    "YEAR"
                    if "year" in grain_terms
                    else "MONTH"
                    if "month" in grain_terms
                    else "DAY"
                    if "day" in grain_terms
                    else str(column.get("date_grain") or "").upper() or None
                )
                aggregation = None
        elif is_case_count:
            identifier_candidates = {}
            for name, field in fields.items():
                label_terms = _semantic_terms(
                    " ".join(field["labels"]) + " " + name
                )
                identifier_score = len(
                    label_terms
                    & {"identifier", "document", "policy", "number", "code", "record"}
                )
                if identifier_score:
                    identifier_candidates[name] = identifier_score
            if not identifier_candidates:
                return False
            best = max(identifier_candidates.values())
            winners = [
                name for name, score in identifier_candidates.items()
                if score == best
            ]
            if len(winners) != 1:
                return False
            field_name = winners[0]
            aggregation = "COUNT"
            date_grain = None
        elif alias_lower in {"year", "month", "day", "date", "period"} or (
            date_grain
            and isinstance(field_name, str)
            and field_name.upper() in fields
            and _is_date_field(field_name.upper(), fields[field_name.upper()])
        ):
            candidates = {
                name: field
                for name, field in fields.items()
                if _is_date_field(name, field)
            }
            if not candidates:
                return False
            scores = {
                name: sum(
                    len(_semantic_terms(label) & context_terms)
                    for label in field["labels"]
                )
                + len(_semantic_terms(" ".join(field["lineage"])) & context_terms)
                for name, field in candidates.items()
            }
            best = max(scores.values())
            winners = [name for name, score in scores.items() if score == best]
            if best == 0 or len(winners) != 1:
                return False
            field_name = winners[0]
            if not date_grain:
                date_grain = (
                    "MONTH"
                    if re.search(r"\b(month|monthly)\b", prompt, re.IGNORECASE)
                    else "YEAR"
                    if alias_lower == "year"
                    else "DAY"
                    if alias_lower in {"day", "date"}
                    else None
                )
            aggregation = None
        else:
            label_scores = {}
            primary_request = re.split(
                r"\bcolumns?\b", prompt, maxsplit=1, flags=re.IGNORECASE
            )[0]
            primary_terms = _semantic_terms(primary_request)
            for name, field in fields.items():
                if _is_date_field(name, field):
                    continue
                score = 0
                for label in field["labels"]:
                    label_terms = _semantic_terms(label)
                    if label_terms and label_terms <= context_terms:
                        score += len(label_terms) + 1
                    if label_terms and label_terms <= primary_terms:
                        score += (len(label_terms) + 1) * 2
                if name in context_terms:
                    score += 1
                if score:
                    label_scores[name] = score
            if not label_scores:
                if isinstance(field_name, str) and field_name.upper() in fields:
                    return False
                return False
            best = max(label_scores.values())
            winners = [
                name for name, score in label_scores.items() if score == best
            ]
            if len(winners) != 1:
                return False
            field_name = winners[0]
            if aggregation is not None:
                aggregation = str(aggregation).upper()
            date_grain = None

        if not isinstance(field_name, str) or field_name.upper() not in fields:
            return False
        resolved.append(
            {
                "field": field_name.upper(),
                "aggregation": aggregation,
                "alias": alias,
                "date_grain": str(date_grain).upper() if date_grain else None,
            }
        )

    procedure_names = {
        fields[column["field"]]["procedure"] for column in resolved
    }
    if len(procedure_names) != 1:
        return False

    plan["select_columns"] = resolved
    plan["status"] = "ready"
    plan["group_by"] = [
        column["field"]
        for column in resolved
        if column["aggregation"] is None
    ]
    aggregated = [column for column in resolved if column["aggregation"]]
    plan["measure_field"] = aggregated[0]["field"] if aggregated else resolved[0]["field"]
    plan["aggregation"] = (
        aggregated[0]["aggregation"] if aggregated else "COUNT_DISTINCT"
    )
    plan["message"] = "Requested columns matched to annotated package fields."
    return True


def _build_sql(plan: Dict[str, Any], metadata: PackageMetadata) -> Dict[str, Any]:
    fields = metadata.field_catalog()
    selected_columns = plan.get("select_columns") or []
    field_names = (
        [column["field"] for column in selected_columns]
        if selected_columns
        else [plan["measure_field"], *plan["group_by"]]
    )
    procedure_names = {fields[field]["procedure"] for field in field_names}
    if len(procedure_names) != 1:
        raise ValueError("Selected fields do not belong to the same package procedure.")
    procedure = next(iter(procedure_names))

    base_query = metadata.cursor_query(procedure)
    selected_groups = plan["group_by"]
    result_heading = None
    result_values_heading = None
    display_columns = {}
    value_sql = None
    if (
        plan["aggregation"] == "COUNT_DISTINCT"
        and not selected_columns
        and not selected_groups
    ):
        result_heading, result_values_heading = _distinct_dimension_headings(
            plan["user_prompt"],
            plan["measure_field"],
            metadata,
        )
        value_sql = (
            f"SELECT DISTINCT QUERY_DATA.{plan['measure_field']}\n"
            f"FROM (\n{base_query}\n) QUERY_DATA\n"
            f"WHERE QUERY_DATA.{plan['measure_field']} IS NOT NULL\n"
            f"ORDER BY QUERY_DATA.{plan['measure_field']}"
        )
        sql = (
            f"SELECT COUNT(DISTINCT QUERY_DATA.{plan['measure_field']}) "
            "AS DISTINCT_COUNT\n"
            f"FROM (\n{base_query}\n) QUERY_DATA"
        )
    elif selected_columns:
        select_items = []
        group_expressions = []
        for column in selected_columns:
            field_expression = f"QUERY_DATA.{column['field']}"
            date_grain = column.get("date_grain")
            if date_grain:
                trunc_grain = {"YEAR": "YYYY", "MONTH": "MM", "DAY": "DD"}[
                    date_grain
                ]
                expression = f"TRUNC({field_expression}, '{trunc_grain}')"
                select_expression = (
                    f"TO_CHAR({expression}, 'MON YYYY', "
                    "'NLS_DATE_LANGUAGE=ENGLISH')"
                    if date_grain == "MONTH"
                    else f"TO_CHAR({expression}, '{trunc_grain}')"
                )
                group_expressions.append(expression)
            elif column.get("aggregation"):
                select_expression = (
                    f"COUNT(DISTINCT {field_expression})"
                    if column["aggregation"] == "COUNT_DISTINCT"
                    else f"{column['aggregation']}({field_expression})"
                )
            else:
                select_expression = field_expression
                group_expressions.append(field_expression)
            select_items.append(f'{select_expression} AS "{column["alias"]}"')

        sql = (
            "SELECT "
            + ", ".join(select_items)
            + "\nFROM (\n"
            + base_query
            + "\n) QUERY_DATA"
        )
        if group_expressions:
            sql += "\nGROUP BY " + ", ".join(group_expressions)
        date_order = next(
            (
                column
                for column in selected_columns
                if column.get("date_grain")
            ),
            None,
        )
        if date_order:
            grain = {"YEAR": "YYYY", "MONTH": "MM", "DAY": "DD"}[
                date_order["date_grain"]
            ]
            sql += f"\nORDER BY TRUNC(QUERY_DATA.{date_order['field']}, '{grain}')"
        display_columns = {}
    else:
        select_items = []
        group_expressions = []
        group_by_grains = plan.get("group_by_grains", {})
        for field in selected_groups:
            grain = group_by_grains.get(field)
            if grain:
                trunc_grain = {"YEAR": "YYYY", "MONTH": "MM", "DAY": "DD"}[grain]
                expression = f"TRUNC(QUERY_DATA.{field}, '{trunc_grain}')"
                select_items.append(
                    f"TO_CHAR({expression}, '{trunc_grain}') AS "
                    f'"{grain.title()}"'
                )
                group_expressions.append(expression)
                display_columns[grain.title()] = grain.title()
            else:
                select_items.append(f"QUERY_DATA.{field}")
                group_expressions.append(f"QUERY_DATA.{field}")
        measure_expression = (
            f"COUNT(DISTINCT QUERY_DATA.{plan['measure_field']})"
            if plan["aggregation"] == "COUNT_DISTINCT"
            else f"{plan['aggregation']}(QUERY_DATA.{plan['measure_field']})"
        )
        select_items.append(
            f"{measure_expression} AS {plan['measure_field']}"
        )
        sql = (
            "SELECT "
            + ", ".join(select_items)
            + "\nFROM (\n"
            + base_query
            + "\n) QUERY_DATA"
        )
        if group_expressions:
            sql += "\nGROUP BY " + ", ".join(group_expressions)
        display_columns = {
            field: _field_display_label(plan["user_prompt"], fields[field])
            for field in [*selected_groups, plan["measure_field"]]
        }

    if not re.match(r"^\s*SELECT\b", sql, re.IGNORECASE) or ";" in sql:
        raise ValueError("Generated query failed the read-only SQL check.")

    parameters = _parse_request_date_range(plan["user_prompt"])
    uncommented_query = re.sub(r"--[^\n]*|/\*[\s\S]*?\*/", "", base_query)
    bind_parameters = [
        parameter.upper()
        for parameter in re.findall(
            r":([A-Z][A-Z0-9_$#]*)",
            uncommented_query,
            re.IGNORECASE,
        )
    ]
    if not parameters and any(name in bind_parameters for name in ("P_SDATE", "P_EDATE")):
        if plan["aggregation"] == "COUNT_DISTINCT":
            parameters = {
                "P_SDATE": "0001-01-01",
                "P_EDATE": "9999-12-31 23:59:59",
            }
        else:
            return {
                **plan,
                "status": "clarify",
                "sql": None,
                "parameters": {},
                "required_parameters": sorted(set(bind_parameters)),
                "message": "Please specify the requested date range.",
            }
    required_parameters = sorted(set(bind_parameters) - set(parameters))

    return {
        **plan,
        "status": "ready",
        "sql": sql,
        "value_sql": value_sql,
        "result_heading": result_heading,
        "result_values_heading": result_values_heading,
        "display_columns": display_columns,
        "measure_heading": (
            display_columns.get(plan["measure_field"])
            if plan["aggregation"] != "COUNT_DISTINCT"
            else None
        ),
        "operation": (
            "distinct_values"
            if plan["aggregation"] == "COUNT_DISTINCT"
            and not selected_columns
            and not selected_groups
            else plan.get("operation")
        ),
        "parameters": parameters or {},
        "required_parameters": sorted(set(required_parameters)),
    }


def _apply_seeded_semantics(
    plan: Dict[str, Any],
    prompt: str,
    metadata: PackageMetadata,
) -> bool:
    """Apply confirmed package concepts to measure and date-group choices."""
    measure_concept = _match_seeded_concept(
        prompt,
        metadata,
        role="measure",
    )
    if measure_concept is not None:
        plan["measure_field"] = measure_concept["field"]
        plan["aggregation"] = measure_concept["aggregation"]
        if plan.get("status") == "clarify":
            plan["status"] = "ready"
            plan["message"] = "Measure resolved from the package business-concept seed."

    registration_concept = _match_seeded_concept(
        prompt,
        metadata,
        role="dimension",
    )
    requested_grain = _requested_date_group_grain(prompt)
    if registration_concept is not None and requested_grain is not None:
        fields = metadata.field_catalog()
        date_field = registration_concept["field"]
        if date_field not in fields:
            return False
        groups = plan.get("group_by")
        if not isinstance(groups, list):
            return False
        date_groups = [
            field
            for field in groups
            if isinstance(field, str)
            and field.upper() in fields
            and _is_date_field(field.upper(), fields[field.upper()])
        ]
        groups = [
            field
            for field in groups
            if field not in date_groups and field.upper() != date_field
        ]
        groups.append(date_field)
        plan["group_by"] = groups
        plan["group_by_grains"] = {date_field: requested_grain}

    return measure_concept is not None or registration_concept is not None


def _match_annotated_field_alias(
    alias: str,
    metadata: PackageMetadata,
) -> Optional[str]:
    alias_terms = _semantic_terms(alias)
    if not alias_terms:
        return None

    scores = {}
    for name, field in metadata.field_catalog().items():
        candidates = [name.replace("_", " "), *field["labels"]]
        matching_scores = []
        for candidate in candidates:
            candidate_terms = _semantic_terms(candidate)
            if alias_terms <= candidate_terms:
                matching_scores.append(
                    (len(alias_terms), int(alias_terms == candidate_terms))
                )
        if matching_scores:
            scores[name] = max(matching_scores)

    if not scores:
        return None
    best_score = max(scores.values())
    winners = [name for name, score in scores.items() if score == best_score]
    return winners[0] if len(winners) == 1 else None


def _seeded_output_plan(
    prompt: str,
    metadata: PackageMetadata,
) -> Optional[Dict[str, Any]]:
    """Build a query without Ollama when every requested column maps uniquely."""
    aliases = _requested_output_aliases(prompt)
    if not aliases:
        return None

    fields = metadata.field_catalog()
    selected_columns = []
    for alias in aliases:
        explanation = _column_explanation(prompt, alias)
        concept = _match_seeded_concept(alias + " " + explanation, metadata)
        if concept is None and alias.strip().lower() in {
            "year",
            "month",
            "day",
            "date",
            "period",
        }:
            concept = _match_seeded_concept(
                prompt,
                metadata,
                role="dimension",
            )

        if concept is not None:
            field_name = concept["field"]
            aggregation = concept.get("aggregation")
            if concept["role"] == "dimension":
                grain_terms = _semantic_terms(alias + " " + explanation)
                date_grain = (
                    "YEAR"
                    if "year" in grain_terms
                    else "MONTH"
                    if "month" in grain_terms
                    else "DAY"
                    if "day" in grain_terms
                    else _requested_date_group_grain(prompt)
                )
                aggregation = None
            else:
                date_grain = None
        else:
            field_name = _match_annotated_field_alias(alias, metadata)
            if field_name is None:
                return None
            aggregation = None
            date_grain = None
            if _is_date_field(field_name, fields[field_name]):
                grain_terms = _semantic_terms(alias + " " + explanation)
                date_grain = (
                    "YEAR"
                    if "year" in grain_terms
                    else "MONTH"
                    if "month" in grain_terms
                    else "DAY"
                    if "day" in grain_terms
                    else _requested_date_group_grain(prompt)
                )

        if field_name not in fields:
            return None
        selected_columns.append(
            {
                "field": field_name,
                "aggregation": aggregation,
                "alias": alias.strip(),
                "date_grain": date_grain,
            }
        )

    measures = [column for column in selected_columns if column["aggregation"]]
    if not measures:
        return None
    procedure_names = {
        fields[column["field"]]["procedure"] for column in selected_columns
    }
    if len(procedure_names) != 1:
        return None

    plan = {
        "status": "ready",
        "measure_field": measures[0]["field"],
        "aggregation": measures[0]["aggregation"],
        "group_by": [
            column["field"]
            for column in selected_columns
            if column["aggregation"] is None
        ],
        "select_columns": selected_columns,
        "user_prompt": prompt,
        "message": "Requested columns resolved from the live package concept catalogue.",
    }
    return _build_sql(plan, metadata)


def _seeded_grouped_plan(
    prompt: str,
    metadata: PackageMetadata,
) -> Optional[Dict[str, Any]]:
    """Resolve simple measures and explicitly requested groupings from the seed."""
    if _requested_output_aliases(prompt):
        return None
    measure = _match_seeded_concept(prompt, metadata, role="measure")
    if measure is None:
        return None

    fields = metadata.field_catalog()
    groups = []
    group_by_grains = {}
    registration = _match_seeded_concept(
        prompt,
        metadata,
        role="dimension",
    )
    requested_grain = _requested_date_group_grain(prompt)
    if registration is not None and requested_grain is not None:
        groups.append(registration["field"])
        group_by_grains[registration["field"]] = requested_grain

    grouping_patterns = (
        r"\s*[- ]?wise\b",
        r"\b(?:by|per|across|within)\s+(?:the\s+)?",
        r"\bgroup(?:ed)?\s+by\s+(?:the\s+)?",
        r"\b(?:each|every)\s+",
    )
    for name, field in fields.items():
        if name == measure["field"] or _is_date_field(name, field):
            continue
        aliases = [name.replace("_", " "), *field["labels"]]
        requested = False
        for alias in aliases:
            alias_terms = _semantic_terms(alias)
            if registration is not None and any(
                alias_terms <= _semantic_terms(seed_alias)
                for seed_alias in registration["aliases"]
            ):
                continue
            alias_words = alias.split()
            last_word = alias_words[-1]
            if (
                len(last_word) > 1
                and last_word.lower().endswith("y")
                and last_word[-2].lower() not in "aeiou"
            ):
                alias_words[-1] = re.escape(last_word[:-1]) + r"(?:y|ies)"
            elif last_word.lower().endswith(("s", "x", "z", "ch", "sh")):
                alias_words[-1] = re.escape(last_word) + r"(?:es)?"
            else:
                alias_words[-1] = re.escape(last_word) + r"s?"
            alias_pattern = r"\s+".join(
                [*(re.escape(word) for word in alias_words[:-1]), alias_words[-1]]
            )
            patterns = [
                rf"\b{alias_pattern}{grouping_patterns[0]}",
                rf"{grouping_patterns[1]}{alias_pattern}\b",
                rf"{grouping_patterns[2]}{alias_pattern}\b",
                rf"{grouping_patterns[3]}{alias_pattern}\b",
            ]
            if any(re.search(pattern, prompt, re.IGNORECASE) for pattern in patterns):
                requested = True
                break
        if requested:
            groups.append(name)

    procedure_names = {
        fields[field]["procedure"] for field in [measure["field"], *groups]
    }
    if len(procedure_names) != 1:
        return None

    plan = {
        "status": "ready",
        "measure_field": measure["field"],
        "aggregation": measure["aggregation"],
        "group_by": groups,
        "group_by_grains": group_by_grains,
        "select_columns": [],
        "user_prompt": prompt,
        "message": "Business measure and groupings resolved from the package concept catalogue.",
    }
    return _build_sql(plan, metadata)


def _seeded_direct_count_plan(
    prompt: str,
    metadata: PackageMetadata,
) -> Optional[Dict[str, Any]]:
    prompt_terms = _semantic_terms(prompt)
    matches = []
    for concept in metadata.direct_concepts:
        for alias in concept["aliases"]:
            alias_terms = _semantic_terms(alias)
            if alias_terms and alias_terms <= prompt_terms:
                matches.append((len(alias_terms), concept))
    if not matches:
        return None

    best_score = max(score for score, _ in matches)
    concepts = {
        concept["id"]: concept
        for score, concept in matches
        if score == best_score
    }
    if len(concepts) != 1:
        return None

    concept = next(iter(concepts.values()))
    return {
        "status": "ready",
        "operation": "direct_count",
        "measure_field": concept["field"],
        "aggregation": concept["aggregation"],
        "group_by": [],
        "select_columns": [],
        "user_prompt": prompt,
        "message": "Count resolved from the package business-concept catalogue.",
        "sql": (
            f'SELECT {concept["aggregation"]}({concept["field"]}) '
            f'AS "{concept["heading"]}"\n'
            f'FROM {concept["table"]}'
        ),
        "parameters": {},
        "required_parameters": [],
        "display_columns": {},
        "measure_heading": concept["heading"],
    }


def _seeded_lookup_plan(
    prompt: str,
    metadata: PackageMetadata,
) -> Optional[Dict[str, Any]]:
    prompt_terms = _semantic_terms(prompt)
    lookup_candidates = []
    for concept in metadata.lookup_concepts:
        intent_match = any(
            _semantic_terms(alias) <= prompt_terms
            for alias in concept["aliases"]
        )
        input_match = any(
            _semantic_terms(alias) <= prompt_terms
            for alias in concept["input"]["aliases"]
        )
        if intent_match and input_match:
            lookup_candidates.append(concept)
    if len(lookup_candidates) != 1:
        return None

    concept = lookup_candidates[0]
    values = re.findall(concept["input"]["pattern"], prompt)
    if len(values) != 1:
        return {
            "status": "clarify",
            "operation": "lookup",
            "message": (
                "Please provide exactly one valid CNIC number for the policy lookup."
            ),
            "sql": None,
            "parameters": {},
            "required_parameters": [],
        }

    source = concept["from"]
    join_sql = "\n".join(
        (
            f'{join["type"]} {join["table"]} {join["alias"]} '
            f'ON {join["left"]} = {join["right"]}'
        )
        for join in concept["joins"]
    )
    predicates = [
        f'{concept["input"]["field"]} = :{concept["input"]["bind"]}',
        *(
            f'{condition["field"]} {condition["operator"]} '
            f':{condition["bind"]}'
            for condition in concept["conditions"]
        ),
    ]
    sql = (
        f'SELECT DISTINCT {concept["output"]["field"]} '
        f'AS "{concept["output"]["heading"]}"\n'
        f'FROM {source["table"]} {source["alias"]}\n'
        f'{join_sql}\n'
        f'WHERE {" AND ".join(predicates)}'
    )
    return {
        "status": "ready",
        "operation": "lookup",
        "measure_field": concept["output"]["field"],
        "aggregation": "LOOKUP",
        "group_by": [],
        "select_columns": [],
        "user_prompt": prompt,
        "message": "Policy lookup resolved from the package business-concept catalogue.",
        "sql": sql,
        "parameters": {concept["input"]["bind"]: values[0]},
        "required_parameters": [],
        "display_columns": {},
        "empty_result_message": "Policy does not exist",
    }


def interpret_prompt(
    user_prompt: str,
    metadata: PackageMetadata,
    model: Optional[str] = None,
    ollama_url: Optional[str] = None,
    cancellation: Optional[CancellationToken] = None,
) -> Dict[str, Any]:
    """Interpret a prompt with local Ollama and compose SQL from package source."""
    prompt = user_prompt.strip()
    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    cash_value_lookup = _cash_value_lookup(prompt, metadata)
    if cash_value_lookup is not None:
        return cash_value_lookup

    seeded_lookup = _seeded_lookup_plan(prompt, metadata)
    if seeded_lookup is not None:
        return seeded_lookup

    direct_count = _seeded_direct_count_plan(prompt, metadata)
    if direct_count is not None:
        return direct_count

    distinct_count = _distinct_dimension_count_plan(prompt, metadata)
    if distinct_count is not None:
        return distinct_count

    seeded_output = _seeded_output_plan(prompt, metadata)
    if seeded_output is not None:
        return seeded_output
    seeded_grouped = _seeded_grouped_plan(prompt, metadata)
    if seeded_grouped is not None:
        return seeded_grouped

    if cancellation is not None:
        cancellation.check()
    if _is_information_request(
        prompt,
        ollama_url=ollama_url,
        cancellation=cancellation,
    ):
        return {
            "status": "informational",
            "measure_field": None,
            "aggregation": None,
            "group_by": [],
            "select_columns": [],
            "user_prompt": prompt,
            "message": metadata.describe_capabilities(),
            "sql": None,
            "parameters": {},
            "required_parameters": [],
        }

    base_url = (ollama_url or os.environ.get("OLLAMA_URL") or DEFAULT_OLLAMA_URL).rstrip("/")
    model_name = model or os.environ.get("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
    request_body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"PACKAGE METADATA:\n{metadata.prompt_context(prompt)}\n\n"
                    f"USER REQUEST:\n{prompt}"
                ),
            },
        ],
        "format": RESPONSE_FORMAT,
        "stream": False,
        "keep_alive": "10m",
        "options": {"temperature": 0, "num_predict": 256},
    }
    request = Request(
        f"{base_url}/api/chat",
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        payload = _read_ollama_json(request, cancellation, timeout=300)
    except HTTPError as error:
        if cancellation is not None:
            cancellation.check()
        raise RuntimeError(f"Local Ollama returned HTTP {error.code}.") from error
    except (URLError, TimeoutError) as error:
        if cancellation is not None:
            cancellation.check()
        raise RuntimeError(
            f"Local Ollama did not respond at {base_url}; check Ollama and model performance."
        ) from error

    content = payload.get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("Local Ollama response did not contain message content.")

    plan = _read_plan(content, metadata)
    plan["user_prompt"] = prompt
    if plan["status"] == "informational":
        return {
            **plan,
            "sql": None,
            "parameters": {},
            "required_parameters": [],
        }
    if plan.get("select_columns"):
        if _resolve_output_columns(plan, prompt, metadata):
            return _build_sql(plan, metadata)
        return {
            **plan,
            "status": "clarify",
            "sql": None,
            "parameters": {},
            "required_parameters": [],
            "message": (
                "I could not safely map every requested output column to the "
                "annotated package fields and seeded business concepts."
            ),
        }
    _apply_seeded_semantics(plan, prompt, metadata)
    if plan["status"] != "ready":
        if _resolve_distinct_dimension_count(plan, prompt, metadata):
            return _build_sql(plan, metadata)
        if plan["status"] == "clarify" and _resolve_annotated_measure(plan, metadata):
            return _build_sql(plan, metadata)
        return {**plan, "sql": None, "parameters": {}, "required_parameters": []}
    return _build_sql(plan, metadata)
