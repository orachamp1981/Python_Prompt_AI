"""
BIEL Light Path — provenance-driven query routing.

The full procedure SQL is expensive: it joins transaction tables, a daily
business table and two DB links. Most simple questions do not need any of
that. This module decides, deterministically, whether a question can be
answered from a small whitelist of base tables — and builds that SQL from
a template.

The LLM is never asked to pick tables or joins. It only resolves business
vocabulary to canonical column names. Everything structural comes from
parsing the proven procedure SQL, so the light path cannot invent a join
that the business logic does not already contain.

Two inputs drive it:
  1. Column provenance  — auto-derived from the procedure source.
  2. Table whitelist    — declared by the DBA via @light_* annotations in
                          the package, or a sidecar JSON file.
"""

import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

STORE_DIR = Path("extracted_packages")

# Functions that are safe to re-evaluate in a light query. Anything outside
# this set is treated as opaque business logic and forces the full path.
BUILTIN_FUNCS = {
    "NVL", "NVL2", "DECODE", "TO_DATE", "TO_CHAR", "TO_NUMBER", "TRUNC",
    "ROUND", "ABS", "SIGN", "MOD", "POWER", "SQRT", "CEIL", "FLOOR",
    "UPPER", "LOWER", "INITCAP", "TRIM", "LTRIM", "RTRIM", "SUBSTR",
    "INSTR", "LENGTH", "REPLACE", "LPAD", "RPAD", "CONCAT",
    "SYSDATE", "GREATEST", "LEAST", "COALESCE", "NULLIF",
    "SUM", "COUNT", "AVG", "MIN", "MAX",
    "EXTRACT", "ADD_MONTHS", "MONTHS_BETWEEN", "LAST_DAY", "NEXT_DAY",
}

CLAUSE_STOPS = (
    "WHERE", "GROUP BY", "HAVING", "ORDER BY",
    "UNION", "MINUS", "INTERSECT",
    "CONNECT BY", "START WITH",
)

AGGREGATE_FUNCS = {"SUM", "AVG", "MIN", "MAX", "COUNT"}


# ============================================================
# SQL TEXT HELPERS
# ============================================================

def mask_sql(sql: str) -> Tuple[str, str]:
    """
    Returns two length-preserving copies of the SQL:
      blanked — comments and string literals replaced by spaces
      clean   — comments replaced by spaces, string literals kept

    Identical lengths mean an index found in one is valid in the other,
    which is what lets us locate structure in `blanked` and then slice
    the real text out of `clean`.
    """
    blanked = list(sql)
    clean = list(sql)
    i = 0
    n = len(sql)

    while i < n:
        if sql[i:i + 2] == "--":
            j = i
            while j < n and sql[j] != "\n":
                blanked[j] = " "
                clean[j] = " "
                j += 1
            i = j
            continue

        if sql[i:i + 2] == "/*":
            j = i
            while j < n and sql[j:j + 2] != "*/":
                blanked[j] = " "
                clean[j] = " "
                j += 1
            if j < n:
                blanked[j] = blanked[j + 1] = " "
                clean[j] = clean[j + 1] = " "
                j += 2
            i = j
            continue

        if sql[i] == "'":
            j = i
            while j < n:
                blanked[j] = " "
                if sql[j] == "'":
                    if j + 1 < n and sql[j + 1] == "'":
                        blanked[j + 1] = " "
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            i = j
            continue

        i += 1

    return "".join(blanked), "".join(clean)


def _keyword_at(upper: str, pos: int, keyword: str) -> bool:
    if not upper.startswith(keyword, pos):
        return False
    before_ok = pos == 0 or not (upper[pos - 1].isalnum() or upper[pos - 1] == "_")
    after = pos + len(keyword)
    after_ok = after >= len(upper) or not (upper[after].isalnum() or upper[after] == "_")
    return before_ok and after_ok


def scan_clause_end(blanked: str, start: int) -> Tuple[int, Optional[str]]:
    """
    Walks forward from `start` and returns where the current clause ends,
    stopping only at a keyword sitting at the same parenthesis depth.
    """
    upper = blanked.upper()
    n = len(blanked)
    depth = 0
    i = start

    while i < n:
        ch = blanked[i]

        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                return i, ")"
            depth -= 1
        elif ch == ";" and depth == 0:
            return i, ";"
        elif depth == 0:
            for kw in CLAUSE_STOPS:
                if _keyword_at(upper, i, kw):
                    return i, kw
        i += 1

    return n, None


def split_top_level(text: str, delimiter: str = ",") -> List[Tuple[str, int, int]]:
    """
    Splits on a delimiter that sits at parenthesis depth zero.
    Returns (slice, start, end) so callers can map back into a
    parallel string of the same length.
    """
    blanked, _ = mask_sql(text)
    parts = []
    start = 0
    depth = 0

    for i, ch in enumerate(blanked):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == delimiter and depth == 0:
            if text[start:i].strip():
                parts.append((text[start:i], start, i))
            start = i + 1

    if text[start:].strip():
        parts.append((text[start:], start, len(text)))

    return parts


# ============================================================
# CONFIGURATION
# ============================================================

class LightPathConfig:
    """The DBA-declared whitelist. Nothing here is inferred."""

    def __init__(
        self,
        enabled: bool,
        base_alias: str,
        base_table: str,
        count_key: str,
        allowed_tables: Set[str],
        blocked_tables: Set[str],
        mandatory_filters: List[str],
        column_types: Optional[Dict[str, str]] = None,
    ):
        self.enabled = enabled
        self.base_alias = base_alias.upper()
        self.base_table = base_table.upper()
        self.count_key = count_key
        self.allowed_tables = {t.upper() for t in allowed_tables}
        self.blocked_tables = {t.upper() for t in blocked_tables}
        self.mandatory_filters = mandatory_filters
        self.column_types = column_types or {}

    def table_allowed(self, table: str) -> bool:
        t = table.upper()
        if t in self.blocked_tables:
            return False
        if self.allowed_tables:
            return t in self.allowed_tables
        # No explicit allow-list: permit anything local and not blocked.
        return "@" not in t


def parse_light_annotations(source: str) -> Optional[Dict[str, Any]]:
    """
    Reads @light_* annotations out of the procedure source. Grammar:

        -- @light_base: T1 = EPF_UW_DOCUMENT
        -- @light_count_key: T1.DOCUMENT_ID_PK
        -- @light_allow: EPF_UW_EMPLOYEES, EPF_GEN_ST_PLAN
        -- @light_block: BUSINESS_010226, AMMM_AGENT_MASTER@PENTA_FTPROD
        -- @light_filter: T1.PARTICIPANT_TYPE = 'I'
        -- @light_filter: T1.COMPANYID_FK = :p_company_code

    @light_filter may repeat. Returns None when no base is declared,
    which means the package has not opted in to the light path.
    """
    def find_all(key: str) -> List[str]:
        return [
            m.group(1).strip()
            for m in re.finditer(
                rf"--\s*@{key}\s*:\s*(.+)$", source, re.IGNORECASE | re.MULTILINE
            )
        ]

    def split_list(values: List[str]) -> Set[str]:
        out = set()
        for value in values:
            out.update(v.strip() for v in value.split(",") if v.strip())
        return out

    base_entries = find_all("light_base")
    if not base_entries:
        return None

    base_raw = base_entries[0]
    m = re.match(r"^(\w+)\s*=\s*([\w$#@.]+)$", base_raw.strip())
    if m:
        base_alias, base_table = m.group(1), m.group(2)
    else:
        base_alias, base_table = base_raw.strip(), ""

    count_keys = find_all("light_count_key")

    return {
        "enabled": True,
        "base_alias": base_alias.upper(),
        "base_table": base_table.upper(),
        "count_key": count_keys[0] if count_keys else None,
        "allowed_tables": split_list(find_all("light_allow")),
        "blocked_tables": split_list(find_all("light_block")),
        "mandatory_filters": find_all("light_filter"),
    }


def load_sidecar_config(package_name: str) -> Optional[Dict[str, Any]]:
    """
    Sidecar config, used when the annotations have not been deployed into
    the package yet. Same shape as parse_light_annotations() output.
    """
    path = STORE_DIR / f"{package_name.upper()}.lightpath.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw["allowed_tables"] = set(raw.get("allowed_tables", []))
    raw["blocked_tables"] = set(raw.get("blocked_tables", []))
    return raw


def resolve_config(
    proc_source: str,
    package_name: str,
    column_types: Optional[Dict[str, str]] = None,
) -> Optional[LightPathConfig]:
    raw = parse_light_annotations(proc_source) or load_sidecar_config(package_name)
    if not raw or not raw.get("enabled"):
        return None
    if not raw.get("count_key"):
        return None
    return LightPathConfig(
        enabled=True,
        base_alias=raw["base_alias"],
        base_table=raw.get("base_table") or "",
        count_key=raw["count_key"],
        allowed_tables=raw.get("allowed_tables") or set(),
        blocked_tables=raw.get("blocked_tables") or set(),
        mandatory_filters=raw.get("mandatory_filters") or [],
        column_types=column_types,
    )


# ============================================================
# PROVENANCE
# ============================================================

class Provenance:
    """What the procedure's base query level actually looks like."""

    def __init__(self):
        self.alias_map: Dict[str, Dict[str, Any]] = {}
        self.columns: Dict[str, Dict[str, Any]] = {}
        self.join_predicates: List[Dict[str, str]] = []
        self.base_filters: List[str] = []
        self.distinct = False

    def tables_for_alias(self, alias: str) -> Set[str]:
        entry = self.alias_map.get(alias.upper())
        if not entry:
            return set()
        return set(entry["tables"])


EQUALITY_RE = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*\.\s*([A-Za-z_]\w*)\s*(\(\+\))?\s*=\s*"
    r"([A-Za-z_]\w*)\s*\.\s*([A-Za-z_]\w*)\s*(\(\+\))?\s*$",
    re.IGNORECASE,
)


def _governing_from(blanked: str, pos: int) -> int:
    """
    Walks backwards from `pos` to the FROM keyword that opens the clause
    containing it. Returns -1 if pos is not directly inside a FROM list
    (for example when it sits in a select list or a nested subquery).
    """
    upper = blanked.upper()
    depth = 0
    i = pos - 1

    while i >= 0:
        ch = blanked[i]
        if ch == ")":
            depth += 1
        elif ch == "(":
            depth -= 1
            if depth < 0:
                return -1
        elif depth == 0 and _keyword_at(upper, i, "FROM"):
            return i
        i -= 1

    return -1


def _find_base_from_block(
    blanked: str, clean: str, base_alias: str
) -> Optional[Tuple[int, int, int]]:
    """
    Locates the FROM clause that physically defines the base alias.

    The procedure nests several SELECT levels and only the innermost one
    lists real tables. Scanning FROM keywords forward is ambiguous — an
    outer FROM body contains the inner one — so we anchor on the alias
    definition itself and walk back to the FROM that governs it.

    Returns (from_keyword_pos, body_start, body_end).
    """
    alias_def = re.compile(
        rf"([A-Za-z_][\w$#]*(?:@[A-Za-z_][\w$#]*)?)\s+{re.escape(base_alias)}\s*(?=,|$)",
        re.IGNORECASE | re.MULTILINE,
    )

    for m in alias_def.finditer(clean):
        from_pos = _governing_from(blanked, m.start())
        if from_pos < 0:
            continue
        body_start = from_pos + len("FROM")
        end, _ = scan_clause_end(blanked, body_start)
        if end <= body_start:
            continue
        return from_pos, body_start, end

    return None


def _parse_from_block(block: str) -> Dict[str, Dict[str, Any]]:
    """
    Turns a FROM clause into alias -> {table, tables, kind, source}.
    Handles both plain tables and inline views / derived tables.
    """
    alias_map: Dict[str, Dict[str, Any]] = {}

    for part, _, _ in split_top_level(block):
        stripped = part.strip()
        if not stripped:
            continue

        blanked_part, clean_part = mask_sql(stripped)

        if blanked_part.startswith("("):
            depth = 0
            close = -1
            for i, ch in enumerate(blanked_part):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        close = i
                        break
            if close < 0:
                continue
            inner = clean_part[1:close]
            alias_tail = clean_part[close + 1:].strip()
            tail_tokens = alias_tail.split()
            if not tail_tokens:
                continue
            alias = tail_tokens[-1]
            alias_map[alias.upper()] = {
                "table": None,
                "tables": _real_tables(inner),
                "kind": "subquery",
                "source": inner,
            }
            continue

        tokens = [t for t in re.split(r"\s+", clean_part.strip()) if t]
        if not tokens:
            continue

        table = tokens[0]
        if len(tokens) == 1:
            alias = table
        elif len(tokens) >= 3 and tokens[-2].upper() == "AS":
            alias = tokens[-1]
        else:
            alias = tokens[-1]

        alias_map[alias.upper()] = {
            "table": table.upper(),
            "tables": {table.upper()},
            "kind": "table",
            "source": clean_part.strip(),
        }

    return alias_map


SQL_KEYWORDS_NOT_TABLES = {
    "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "NULL", "AS", "ON",
    "GROUP", "BY", "ORDER", "HAVING", "DISTINCT", "CASE", "WHEN", "THEN",
    "ELSE", "END", "IN", "EXISTS", "BETWEEN", "LIKE", "IS", "SUM",
    "COUNT", "AVG", "MIN", "MAX", "NVL", "DECODE", "TRUNC", "TO_DATE",
    "TO_CHAR", "TO_NUMBER", "UPPER", "LOWER", "MAX", "MINUS", "UNION",
    "ALL", "ANY", "SOME", "WITH", "JOIN", "LEFT", "RIGHT", "INNER",
    "OUTER", "FULL", "CROSS", "USING", "ASC", "DESC", "OVER",
    "PARTITION", "ROWNUM", "DUAL", "SYSDATE", "NVL2", "COALESCE",
}


def _real_tables(sql_text: str) -> Set[str]:
    """Physical table/view names referenced by a SQL fragment."""
    blanked, _ = mask_sql(sql_text)
    upper = blanked.upper()
    tables = set()

    for m in re.finditer(
        r"\b(?:FROM|JOIN|UPDATE|INTO)\s+([A-Za-z_][\w$#]*(?:@[A-Za-z_][\w$#]*)?)",
        upper,
    ):
        name = m.group(1)
        if name not in SQL_KEYWORDS_NOT_TABLES:
            tables.add(name)

    return tables


def _find_owning_select(blanked: str, from_start: int) -> int:
    """
    Walks backwards from a FROM to the SELECT that owns it, ignoring any
    SELECT belonging to a scalar subquery inside the select list.
    """
    upper = blanked.upper()
    depth = 0
    i = from_start - 1

    while i >= 0:
        ch = blanked[i]
        if ch == ")":
            depth += 1
        elif ch == "(":
            depth -= 1
        elif depth == 0 and _keyword_at(upper, i, "SELECT"):
            return i
        i -= 1

    return -1


def _parse_select_item(item: str, alias_map: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Resolves one select-list entry to: output name, source expression,
    the aliases it touches, and whether it is light-path-safe.
    """
    blanked, clean = mask_sql(item)

    # Strip using the original text's bounds so both strings stay index-aligned.
    lead = len(item) - len(item.lstrip())
    trail = len(item.rstrip())
    blanked_s = blanked[lead:trail]
    clean_s = clean[lead:trail]

    if not blanked_s.strip():
        return None

    # --- Determine the output name ---
    name = None
    expr = clean_s

    simple = re.match(r"^\s*(?:([A-Za-z_]\w*)\s*\.\s*)?([A-Za-z_]\w*)\s*$", blanked_s)
    if simple:
        name = simple.group(2)
    else:
        tail = re.match(r"^(.*?)[\s]+(?:AS\s+)?([A-Za-z_]\w*)\s*$", blanked_s, re.IGNORECASE | re.DOTALL)
        if tail and tail.group(1).strip() and not tail.group(1).strip().endswith("."):
            name = tail.group(2)
            expr = re.sub(r"\s+AS\s+$", "", clean_s[:tail.start(2)].strip(), flags=re.IGNORECASE).strip()
        else:
            return None

    if not name:
        return None

    upper_item = blanked_s.upper()
    is_subquery = re.search(r"\bSELECT\b", upper_item) is not None

    # --- Aliases and tables this item depends on ---
    used_aliases = set()
    opaque_functions = set()

    for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\.\s*([A-Za-z_]\w*)", blanked_s):
        prefix = m.group(1).upper()
        if prefix in alias_map:
            used_aliases.add(prefix)

    # Package/schema-qualified calls are opaque business logic we cannot
    # safely re-evaluate, e.g. PKG_EPF_CALCULATION.SF_GHS_AGE_CALCULATION(...).
    for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\.\s*([A-Za-z_]\w*)\s*\(", blanked_s):
        prefix = m.group(1).upper()
        if prefix not in alias_map and prefix not in BUILTIN_FUNCS:
            opaque_functions.add(prefix)

    for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", blanked_s):
        func = m.group(1).upper()
        if func in BUILTIN_FUNCS or func in SQL_KEYWORDS_NOT_TABLES or func in alias_map:
            continue
        opaque_functions.add(func)

    subquery_tables: Set[str] = _real_tables(item) if is_subquery else set()

    tables: Set[str] = set()
    for alias in used_aliases:
        tables |= alias_map[alias]["tables"]
    tables |= subquery_tables

    return {
        "name": name.upper(),
        "expr": expr,
        "aliases": sorted(used_aliases),
        "tables": sorted(tables),
        "is_subquery": is_subquery,
        "opaque_functions": sorted(opaque_functions),
        "is_literal": not used_aliases and not is_subquery and not opaque_functions,
    }


def extract_provenance(proc_source: str, base_alias: str) -> Optional[Provenance]:
    """
    Parses the base query level of the procedure and records where every
    output column physically comes from.
    """
    blanked, clean = mask_sql(proc_source)

    found = _find_base_from_block(blanked, clean, base_alias)
    if not found:
        return None

    from_kw, from_start, from_end = found
    prov = Provenance()
    prov.alias_map = _parse_from_block(clean[from_start:from_end])

    if base_alias.upper() not in prov.alias_map:
        return None

    # --- WHERE of the base level: joins vs filters ---
    where_pos = -1
    tail = blanked[from_end:from_end + 40].lstrip()
    if tail.upper().startswith("WHERE"):
        where_pos = blanked.index("WHERE", from_end)

    if where_pos >= 0:
        where_body_start = where_pos + len("WHERE")
        where_end, _ = scan_clause_end(blanked, where_body_start)
        where_text = clean[where_body_start:where_end]
        where_blanked = blanked[where_body_start:where_end]

        for pred, _, _ in _split_and_predicates(where_text, where_blanked):
            pred_clean = pred.strip()
            if not pred_clean:
                continue

            pred_blanked = mask_sql(pred_clean)[0].strip()

            m = EQUALITY_RE.match(pred_blanked)
            if m:
                left_alias = m.group(1).upper()
                right_alias = m.group(4).upper()
                if left_alias in prov.alias_map and right_alias in prov.alias_map:
                    prov.join_predicates.append({
                        "raw": pred_clean,
                        "left_alias": left_alias,
                        "right_alias": right_alias,
                        "aliases": sorted({left_alias, right_alias}),
                    })
                    continue

            refs = {a.upper() for a in re.findall(r"\b([A-Za-z_]\w*)\s*\.", pred_blanked)}
            if refs and refs <= set(prov.alias_map):
                prov.base_filters.append(pred_clean)

    # --- SELECT list of the base level ---
    select_pos = _find_owning_select(blanked, from_kw)
    if select_pos < 0:
        return prov

    list_start = select_pos + len("SELECT")
    probe = blanked[list_start:list_start + 40].lstrip().upper()
    if probe.startswith("DISTINCT"):
        prov.distinct = True
        list_start = blanked.index("DISTINCT", list_start) + len("DISTINCT")

    select_text = clean[list_start:from_kw]
    for part, _, _ in split_top_level(select_text):
        parsed = _parse_select_item(part, prov.alias_map)
        if parsed:
            prov.columns[parsed["name"]] = parsed

    return prov


def _split_and_predicates(text: str, blanked: str) -> List[Tuple[str, int, int]]:
    """Splits a WHERE clause on top-level ANDs."""
    parts = []
    depth = 0
    start = 0
    upper = blanked.upper()
    i = 0
    n = len(blanked)

    while i < n:
        ch = blanked[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and _keyword_at(upper, i, "AND"):
            parts.append((text[start:i], start, i))
            start = i + 3
            i += 3
            continue
        i += 1

    parts.append((text[start:], start, n))
    return [p for p in parts if p[0].strip()]


# ============================================================
# ROUTING
# ============================================================

class RouteDecision:
    def __init__(self, allowed: bool, reasons: List[str], aliases: Set[str]):
        self.allowed = allowed
        self.reasons = reasons
        self.aliases = aliases
        self.cautions: List[str] = []
        self.intent: Dict[str, Any] = {}

    def explain(self) -> str:
        return "; ".join(self.reasons) if self.reasons else "ok"


def route_columns(
    column_names: List[str],
    prov: Provenance,
    config: LightPathConfig,
) -> RouteDecision:
    """
    Decides whether every requested column can be served from whitelisted
    tables. A single blocked column sends the whole query to the full path
    — partial answers that silently differ from the report are worse than
    a slow query.
    """
    reasons: List[str] = []
    aliases: Set[str] = {config.base_alias}

    for raw_name in column_names:
        name = (raw_name or "").upper()
        if not name:
            continue

        col = prov.columns.get(name)
        if col is None:
            reasons.append(f"{name}: not produced by the base query level")
            continue

        if col["opaque_functions"]:
            reasons.append(
                f"{name}: depends on business function {', '.join(col['opaque_functions'])}"
            )

        blocked = {t for t in col["tables"] if not config.table_allowed(t)}
        if blocked:
            reasons.append(f"{name}: needs {', '.join(sorted(blocked))}")

        aliases |= set(col["aliases"])

    return RouteDecision(
        allowed=not reasons,
        reasons=reasons,
        aliases=aliases,
    )


def _allowed_aliases(prov: Provenance, config: LightPathConfig) -> Set[str]:
    return {
        alias
        for alias, entry in prov.alias_map.items()
        if all(config.table_allowed(t) for t in entry["tables"])
    }


def _bfs_path(graph: Dict[str, Set[str]], start: str, goal: str) -> Optional[List[str]]:
    if start == goal:
        return [start]
    queue = [[start]]
    seen = {start}
    while queue:
        path = queue.pop(0)
        for nxt in sorted(graph.get(path[-1], ())):
            if nxt in seen:
                continue
            candidate = path + [nxt]
            if nxt == goal:
                return candidate
            seen.add(nxt)
            queue.append(candidate)
    return None


def _join_closure(
    needed_aliases: Set[str],
    prov: Provenance,
    config: LightPathConfig,
) -> Tuple[Optional[Set[str]], List[str], Optional[str]]:
    """
    Builds the smallest set of tables that connects the base table to every
    alias the requested columns need, then returns the procedure's own join
    predicates for that set.

    Bridging tables are pulled in automatically — occupation is reached
    through participant, city through participant — because the procedure
    already declares those links. The graph is restricted to whitelisted
    aliases, so if the only route runs through a blocked table there is no
    path and the query goes to the full procedure.
    """
    permitted = _allowed_aliases(prov, config)
    base = config.base_alias

    if base not in permitted:
        return None, [], "base table is not on the light-path whitelist"

    graph: Dict[str, Set[str]] = {}
    for pred in prov.join_predicates:
        left, right = pred["left_alias"], pred["right_alias"]
        if left in permitted and right in permitted:
            graph.setdefault(left, set()).add(right)
            graph.setdefault(right, set()).add(left)

    keep = {base}
    for alias in sorted(needed_aliases):
        if alias == base:
            continue
        if alias not in permitted:
            return None, [], f"{alias} is not on the light-path whitelist"
        path = _bfs_path(graph, base, alias)
        if path is None:
            return None, [], f"no whitelisted join path from {base} to {alias}"
        keep |= set(path)

    predicates = [
        pred["raw"] for pred in prov.join_predicates
        if set(pred["aliases"]) <= keep
    ]

    # Every kept table must be reachable, otherwise the FROM would produce
    # a cartesian product.
    for alias in keep:
        if alias == base:
            continue
        connected = any(
            alias in set(p["aliases"]) for p in prov.join_predicates
            if set(p["aliases"]) <= keep and p["raw"] in predicates
        )
        if not connected:
            return None, [], f"{alias} would join without a predicate"

    return keep, predicates, None


def build_from_clause(
    aliases: Set[str],
    predicates: List[str],
    prov: Provenance,
    config: LightPathConfig,
) -> Tuple[str, List[str]]:
    """Comma-style FROM plus the procedure's own join predicates verbatim."""
    ordered = [config.base_alias] + sorted(a for a in aliases if a != config.base_alias)

    sources = []
    for alias in ordered:
        entry = prov.alias_map.get(alias)
        if not entry:
            continue
        if entry["kind"] == "table":
            sources.append(
                entry["table"] if alias == entry["table"] else f"{entry['table']} {alias}"
            )
        else:
            sources.append(f"(\n{entry['source']}\n    ) {alias}")

    return "\n         , ".join(sources), list(predicates)


def quote_value(value: Any) -> str:
    if value is None:
        return "NULL"
    text = str(value).strip()
    if re.fullmatch(r"-?\d+(\.\d+)?", text):
        return text
    return "'" + text.replace("'", "''") + "'"


TO_DATE_WRAP_RE = re.compile(
    r"^TO_DATE\s*\(\s*([A-Za-z_]\w*)\s*\.\s*([A-Za-z_]\w*)\s*\)$",
    re.IGNORECASE,
)


def date_filter_expr(col: Dict[str, Any], prov: Provenance, config: LightPathConfig) -> Tuple[str, bool]:
    """
    Returns the best expression for a date range filter on this column,
    plus whether it was simplified.

    The procedure writes TO_DATE(T1.ENTER_DATE) to normalise its output,
    but ENTER_DATE is already a DATE. Using that wrapper in a range filter
    forces an implicit DATE->CHAR->DATE round-trip through the session's
    NLS_DATE_FORMAT and defeats any index on the column. When Oracle
    metadata confirms the column really is a DATE, filter it directly.
    """
    m = TO_DATE_WRAP_RE.match((col["expr"] or "").strip())
    if not m:
        return col["expr"], False

    alias, column = m.group(1).upper(), m.group(2).upper()
    entry = prov.alias_map.get(alias)
    if not entry or not entry.get("table"):
        return col["expr"], False

    dtype = (config.column_types.get(f"{entry['table']}.{column}") or "").upper()
    if dtype.startswith("DATE") or dtype.startswith("TIMESTAMP"):
        return f"{alias}.{column}", True

    return col["expr"], False


# ============================================================
# LIGHT SQL COMPOSITION
# ============================================================

def build_light_sql(
    raw_intent: Dict[str, Any],
    prov: Provenance,
    config: LightPathConfig,
    params: Dict[str, Any],
    label_index: Optional[Dict[str, str]] = None,
    known_columns: Optional[Set[str]] = None,
) -> Tuple[Optional[str], RouteDecision]:
    """
    Resolves a raw model intent against the business vocabulary, then renders
    the light query. Returns (sql, decision); sql is None when the question
    cannot be served from the whitelist.

    Resolution happens in here rather than in the caller on purpose: an
    unrecognised term must never be silently dropped, because the query
    would still run and return a confident-looking but wrong number.
    """
    intent, unresolved = resolve_intent(
        raw_intent,
        label_index if label_index is not None else {},
        known_columns if known_columns is not None else set(prov.columns),
    )

    if unresolved:
        decision = RouteDecision(
            allowed=False,
            reasons=[f"unresolved term: {u}" for u in unresolved],
            aliases=set(),
        )
        decision.intent = intent
        return None, decision

    measure = (intent.get("measure") or "COUNT").upper()
    measure_column = intent.get("measure_column")
    dimensions = [d for d in (intent.get("dimensions") or []) if d]
    filters = intent.get("filters") or []
    date_column = intent.get("date_column")

    needed = list(dimensions)
    if measure_column and measure_column != "*":
        needed.append(measure_column)
    for f in filters:
        if f.get("column"):
            needed.append(f["column"])
    if date_column:
        needed.append(date_column)

    decision = route_columns(needed, prov, config)
    decision.intent = intent
    if not decision.allowed:
        return None, decision

    aliases, predicates, closure_error = _join_closure(decision.aliases, prov, config)
    if closure_error:
        decision.allowed = False
        decision.reasons.append(closure_error)
        return None, decision

    from_sql, where_parts = build_from_clause(aliases, predicates, prov, config)

    # --- SELECT list ---
    select_parts = []
    group_parts = []

    for dim in dimensions:
        col = prov.columns.get(dim.upper())
        if not col:
            continue
        select_parts.append(f"{col['expr']} AS {dim.upper()}")
        group_parts.append(col["expr"])

    count_key = config.count_key

    if measure == "COUNT":
        select_parts.append(f"COUNT(DISTINCT {count_key}) AS ROW_COUNT")
    elif measure in AGGREGATE_FUNCS:
        col = prov.columns.get((measure_column or "").upper())
        if not col:
            decision.allowed = False
            decision.reasons.append(f"measure column {measure_column} unresolved")
            return None, decision
        alias_name = f"TOTAL_{col['name']}" if measure == "SUM" else f"{measure}_{col['name']}"
        select_parts.append(f"{measure}({col['expr']}) AS {alias_name}")
    else:
        # LIST: return the requested columns plus the entity key.
        if not select_parts:
            select_parts.append(f"{count_key} AS ENTITY_KEY")
        for extra in ([measure_column] if measure_column and measure_column != "*" else []):
            col = prov.columns.get(extra.upper())
            if col:
                select_parts.append(f"{col['expr']} AS {col['name']}")

    # --- WHERE ---
    for raw_filter in config.mandatory_filters:
        where_parts.append(_substitute_params(raw_filter, params))

    for f in filters:
        col = prov.columns.get((f.get("column") or "").upper())
        if not col:
            continue
        op = (f.get("op") or "=").upper()
        value = f.get("value")
        if op == "IN" and isinstance(value, (list, tuple)):
            rendered = ", ".join(quote_value(v) for v in value)
            where_parts.append(f"{col['expr']} IN ({rendered})")
        elif op == "IS NULL":
            where_parts.append(f"{col['expr']} IS NULL")
        elif op == "IS NOT NULL":
            where_parts.append(f"{col['expr']} IS NOT NULL")
        else:
            where_parts.append(f"{col['expr']} {op} {quote_value(value)}")

    date_simplified = False
    if date_column and params.get("p_sdate") and params.get("date_explicit"):
        col = prov.columns.get(date_column.upper())
        if col:
            date_expr, date_simplified = date_filter_expr(col, prov, config)
            where_parts.append(
                f"{date_expr} BETWEEN TO_DATE('{params['p_sdate']}', 'YYYY-MM-DD') "
                f"AND TO_DATE('{params['p_edate']}', 'YYYY-MM-DD')"
            )

    sql = "SELECT " + (",\n       ".join(select_parts))
    sql += f"\nFROM {from_sql}"
    if where_parts:
        sql += "\nWHERE " + "\n  AND ".join(where_parts)
    if group_parts and measure != "LIST":
        sql += "\nGROUP BY " + ", ".join(group_parts)
        sql += "\nORDER BY " + ", ".join(str(i + 1) for i in range(len(group_parts)))

    used_tables = sorted({
        t for alias in aliases for t in prov.tables_for_alias(alias)
    })

    if measure == "COUNT":
        effect = "counts will be higher than the full procedure returns"
    elif measure in ("SUM", "AVG", "MIN", "MAX"):
        effect = f"{measure} is not restricted to rows the procedure would keep"
    else:
        effect = "the row set is not restricted to rows the procedure would keep"

    decision.cautions.append(
        "Light path reads only " + ", ".join(used_tables) + ". "
        "It skips the procedure's transaction/business tables and their filters, so "
        + effect + "."
    )

    if not (date_column and params.get("date_explicit")):
        decision.cautions.append("No date bound in the prompt - this is an all-time figure.")

    if date_column and params.get("date_explicit") and not date_simplified:
        col = prov.columns.get(date_column.upper())
        if col and re.search(r"\bTO_DATE\s*\(", col["expr"], re.IGNORECASE):
            decision.cautions.append(
                f"{date_column} is stored as {col['expr']}. That conversion is "
                "NLS-dependent and will not use an index."
            )

    if measure in ("SUM", "AVG") and len(aliases) > 1:
        decision.cautions.append(
            f"{measure} is summed across joined rows. If a policy has multiple "
            "employee rows the amount is counted once per row."
        )

    return sql, decision


def _substitute_params(text: str, params: Dict[str, Any]) -> str:
    out = text
    out = re.sub(r":?p_?company_?code\b", str(params.get("p_company_code", 1)), out, flags=re.IGNORECASE)
    out = re.sub(
        r":?p_?sdate\b",
        f"TO_DATE('{params.get('p_sdate')}', 'YYYY-MM-DD')",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r":?p_?edate\b",
        f"TO_DATE('{params.get('p_edate')}', 'YYYY-MM-DD')",
        out,
        flags=re.IGNORECASE,
    )
    return out


# ============================================================
# VOCABULARY
# ============================================================

def build_label_index(column_tags: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Maps every business synonym (and the canonical name itself) to its
    canonical column. This is what makes "policy number", "document no"
    and DOCUMENT_CODE all resolve to the same thing.
    """
    index: Dict[str, str] = {}
    plsql_keywords = {"BEGIN", "END", "IS", "AS", "DECLARE", "OPEN", "CLOSE", "FETCH"}

    for tag in column_tags or []:
        column = (tag.get("column") or "").upper()
        if not column or column in plsql_keywords:
            continue
        index[column] = column
        for label in tag.get("labels") or []:
            index[label.upper()] = column
        if tag.get("concept"):
            index[tag["concept"].upper()] = column

    return index


def resolve_term(term: str, label_index: Dict[str, str], known_columns: Set[str]) -> Optional[str]:
    """Resolves a free-text term to a canonical column name."""
    if not term:
        return None
    key = term.strip().upper()
    if key in label_index:
        return label_index[key]
    if key in known_columns:
        return key
    singular = key.rstrip("S") if key.endswith("S") else key + "S"
    if singular in label_index:
        return label_index[singular]
    return None


def resolve_intent(
    intent: Dict[str, Any],
    label_index: Dict[str, str],
    known_columns: Set[str],
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Normalises an LLM intent so every column reference is a canonical name
    the provenance map recognises. Unresolvable terms are reported rather
    than guessed at.
    """
    unresolved: List[str] = []
    resolved: Dict[str, Any] = {
        "measure": (intent.get("measure") or "COUNT").upper(),
        "measure_column": None,
        "dimensions": [],
        "filters": [],
        "date_column": None,
    }

    raw_measure = intent.get("measure_column")
    if raw_measure and str(raw_measure).strip() not in ("", "*", "NULL", "NONE"):
        hit = resolve_term(str(raw_measure), label_index, known_columns)
        if hit:
            resolved["measure_column"] = hit
        else:
            unresolved.append(f"measure column '{raw_measure}'")

    for dim in intent.get("dimensions") or []:
        hit = resolve_term(str(dim), label_index, known_columns)
        if hit:
            if hit not in resolved["dimensions"]:
                resolved["dimensions"].append(hit)
        else:
            unresolved.append(f"dimension '{dim}'")

    for f in intent.get("filters") or []:
        hit = resolve_term(str(f.get("column") or ""), label_index, known_columns)
        if hit:
            resolved["filters"].append({
                "column": hit,
                "op": f.get("op") or "=",
                "value": f.get("value"),
            })
        else:
            unresolved.append(f"filter '{f.get('column')}'")

    raw_date = intent.get("date_column")
    if raw_date and str(raw_date).strip().lower() not in ("", "null", "none"):
        hit = resolve_term(str(raw_date), label_index, known_columns)
        if hit:
            resolved["date_column"] = hit
        else:
            unresolved.append(f"date column '{raw_date}'")

    return resolved, unresolved
