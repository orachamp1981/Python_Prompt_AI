# dynamic_engine.py
#
# Generic Oracle Business Intelligence / NL-to-SQL Engine
#
# Architecture:
#   Natural Language
#        ↓
#   Generic Request Planner
#        ↓
#   Static Business SQL Capability Model
#        ↓
#   Capability Matching
#        ↓
#   Query Composition
#        ↓
#   Oracle Validation
#        ↓
#   Result
#
# IMPORTANT:
#   - databases.py is NOT modified.
#   - No business vocabulary mapping is hardcoded.
#   - No prompt-specific rules are used.
#   - Proven SQL from business packages is the primary semantic evidence.
#

import re
import difflib
from collections import defaultdict, Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from databases import get_db_connection


# ============================================================
# CONFIGURATION
# ============================================================

DATABASE_NAME = "PQTEPF"

MAX_SEED_OBJECTS = 12
MAX_SOURCE_OBJECTS = 40
MAX_HISTORY_ROWS = 500
MAX_GRAPH_HOPS = 4
MAX_CANDIDATES = 20

ENABLE_SOURCE_INTELLIGENCE = True
ENABLE_RUNTIME_SQL_HISTORY = True
ENABLE_STATIC_READING_LAYER = True

STATIC_PACKAGE_PREFIX = "PKG_BIEL%"
MAX_STATIC_PACKAGES = 50
MAX_STATIC_PROCEDURES = 200
MAX_STATIC_QUERY_CHARS = 200000

MIN_CAPABILITY_SCORE = 0.35


# ============================================================
# GENERIC LANGUAGE SUPPORT
# ============================================================

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "be", "been", "being",
    "how", "what", "which", "where", "when", "who",
    "many", "much", "do", "does", "did",
    "there", "this", "that", "these", "those",
    "of", "for", "from", "to", "in", "on", "at",
    "with", "without", "by", "per",
    "and", "or", "but", "as", "all", "any", "some",
    "give", "show", "tell", "list", "please", "me",
    "let", "know", "have",
}

OPERATION_WORDS = {
    "count": "COUNT",
    "number": "COUNT",
    "total": "TOTAL",
    "sum": "SUM",
    "average": "AVG",
    "avg": "AVG",
    "minimum": "MIN",
    "minimums": "MIN",
    "min": "MIN",
    "maximum": "MAX",
    "max": "MAX",
}

DIMENSION_WORDS = {
    "wise",
    "per",
    "by",
    "group",
    "grouped",
    "breakdown",
    "break",
}

TIME_WORDS = {
    "today",
    "yesterday",
    "month",
    "year",
    "quarter",
    "week",
    "date",
    "between",
    "during",
}

AGGREGATE_FUNCTIONS = {
    "COUNT",
    "SUM",
    "AVG",
    "MIN",
    "MAX",
}

NUMERIC_TYPES = {
    "NUMBER",
    "FLOAT",
    "BINARY_FLOAT",
    "BINARY_DOUBLE",
    "INTEGER",
    "DECIMAL",
}

DATE_TYPES = {
    "DATE",
    "TIMESTAMP",
    "TIMESTAMP WITH TIME ZONE",
    "TIMESTAMP WITH LOCAL TIME ZONE",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_identifier(value: str) -> str:
    if not value:
        return ""
    value = value.strip().strip('"')
    return value.upper()


def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    return re.findall(r"[a-z0-9_$#]+", text)


def word_forms(word: str) -> set:
    word = word.lower().strip()

    forms = {word}

    if word.endswith("ies") and len(word) > 4:
        forms.add(word[:-3] + "y")

    if word.endswith("es") and len(word) > 4:
        forms.add(word[:-2])

    if word.endswith("s") and len(word) > 3:
        forms.add(word[:-1])

    if word.endswith("ing") and len(word) > 5:
        forms.add(word[:-3])

    return forms


def similarity(a: str, b: str) -> float:
    a = normalize_space(a.lower())
    b = normalize_space(b.lower())

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    if a in b or b in a:
        return 0.90

    return difflib.SequenceMatcher(None, a, b).ratio()


def identifier_parts(identifier: str) -> List[str]:
    identifier = normalize_identifier(identifier)

    identifier = identifier.replace('"', "")
    identifier = re.sub(r"([a-z])([A-Z])", r"\1_\2", identifier)

    return [
        p.lower()
        for p in re.split(r"[_.$#\s]+", identifier)
        if p
    ]


def compact_identifier(identifier: str) -> str:
    return "".join(identifier_parts(identifier))


# ============================================================
# DATABASE HELPERS
# ============================================================

def fetch_rows(
    connection,
    sql: str,
    params: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:

    cursor = connection.cursor()

    try:
        cursor.execute(sql, params or {})

        columns = [
            d[0].upper()
            for d in cursor.description
        ]

        return [
            dict(zip(columns, row))
            for row in cursor.fetchall()
        ]

    finally:
        cursor.close()


def execute_sql(
    connection,
    sql: str,
    params: Optional[Dict[str, Any]] = None
) -> Tuple[List[Tuple], List[str]]:

    cursor = connection.cursor()

    try:
        cursor.execute(sql, params or {})

        columns = [
            d[0].upper()
            for d in cursor.description
        ]

        rows = cursor.fetchall()

        return rows, columns

    finally:
        cursor.close()


# ============================================================
# SQL SOURCE PARSING
# ============================================================

SQL_TABLE_RE = re.compile(
    r"""
    \b
    (?:FROM|JOIN|UPDATE|INTO|DELETE\s+FROM)
    \s+
    ([A-Za-z0-9_$#."@]+)
    """,
    re.I | re.X,
)

ALIAS_RE = re.compile(
    r"""
    \b
    (?:FROM|JOIN)
    \s+
    ([A-Za-z0-9_$#."@]+)
    (?:\s+(?:AS\s+)?([A-Za-z0-9_$#]+))?
    """,
    re.I | re.X,
)

JOIN_EQUALITY_RE = re.compile(
    r"""
    \b
    ([A-Za-z0-9_$#]+)
    \.
    ([A-Za-z0-9_$#]+)
    \s*=\s*
    ([A-Za-z0-9_$#]+)
    \.
    ([A-Za-z0-9_$#]+)
    """,
    re.I | re.X,
)


def mask_sql_comments_and_strings(sql: str) -> str:
    """
    Preserve SQL positions while masking comments and string literals.
    """

    result = list(sql)

    i = 0
    n = len(sql)

    while i < n:

        # -- comment
        if i + 1 < n and sql[i:i + 2] == "--":
            j = i

            while j < n and sql[j] != "\n":
                result[j] = " "
                j += 1

            i = j
            continue

        # /* comment */
        if i + 1 < n and sql[i:i + 2] == "/*":

            j = i + 2

            while j + 1 < n and sql[j:j + 2] != "*/":
                result[j] = " "
                j += 1

            if j + 1 < n:
                result[j] = " "
                result[j + 1] = " "
                j += 2

            i = j
            continue

        # string
        if sql[i] == "'":

            result[i] = " "

            j = i + 1

            while j < n:

                result[j] = " "

                if sql[j] == "'":

                    if j + 1 < n and sql[j + 1] == "'":
                        result[j + 1] = " "
                        j += 2
                        continue

                    j += 1
                    break

                j += 1

            i = j
            continue

        i += 1

    return "".join(result)


def find_matching_parenthesis(sql: str, start: int) -> int:

    depth = 0
    masked = mask_sql_comments_and_strings(sql)

    for i in range(start, len(masked)):

        ch = masked[i]

        if ch == "(":
            depth += 1

        elif ch == ")":
            depth -= 1

            if depth == 0:
                return i

    return -1


def top_level_keyword_positions(
    sql: str,
    keywords: List[str]
) -> Dict[str, int]:

    masked = mask_sql_comments_and_strings(sql)

    positions = {}

    depth = 0
    i = 0

    upper = masked.upper()

    while i < len(masked):

        ch = masked[i]

        if ch == "(":
            depth += 1
            i += 1
            continue

        if ch == ")":
            depth = max(0, depth - 1)
            i += 1
            continue

        if depth == 0:

            for keyword in keywords:

                if upper.startswith(keyword, i):

                    before_ok = (
                        i == 0
                        or not upper[i - 1].isalnum()
                    )

                    after_pos = i + len(keyword)

                    after_ok = (
                        after_pos >= len(upper)
                        or not upper[after_pos].isalnum()
                    )

                    if before_ok and after_ok:
                        positions.setdefault(keyword, i)

        i += 1

    return positions


def split_top_level(text: str, delimiter: str = ",") -> List[str]:

    result = []

    start = 0
    depth = 0

    masked = mask_sql_comments_and_strings(text)

    for i, ch in enumerate(masked):

        if ch == "(":
            depth += 1

        elif ch == ")":
            depth = max(0, depth - 1)

        elif ch == delimiter and depth == 0:

            part = text[start:i].strip()

            if part:
                result.append(part)

            start = i + 1

    final = text[start:].strip()

    if final:
        result.append(final)

    return result


def extract_select_statements(source: str) -> List[str]:
    """
    Generic SELECT extraction from PL/SQL source.

    It intentionally avoids pretending that dynamic EXECUTE IMMEDIATE
    SQL can always be reconstructed safely.
    """

    masked = mask_sql_comments_and_strings(source)

    statements = []

    pattern = re.compile(
        r"\bSELECT\b",
        re.I
    )

    for match in pattern.finditer(masked):

        start = match.start()

        # Avoid SELECTs occurring inside identifiers.
        if start > 0 and masked[start - 1].isalnum():
            continue

        depth = 0
        i = start

        while i < len(masked):

            ch = masked[i]

            if ch == "(":
                depth += 1

            elif ch == ")":
                depth = max(0, depth - 1)

            elif ch == ";" and depth == 0:

                sql = source[start:i].strip()

                if (
                    sql.upper().startswith("SELECT")
                    and len(sql) < MAX_STATIC_QUERY_CHARS
                ):
                    statements.append(sql)

                break

            i += 1

    # Some OPEN cursor FOR SELECT statements do not have a semicolon
    # immediately useful to the extractor. Capture bounded SELECTs.
    if not statements:

        match = re.search(
            r"\bSELECT\b.*",
            source,
            re.I | re.S
        )

        if match:

            candidate = match.group(0).strip()

            if len(candidate) <= MAX_STATIC_QUERY_CHARS:
                statements.append(candidate)

    # Deduplicate normalized SQL.
    seen = set()
    unique = []

    for sql in statements:

        key = normalize_space(sql).upper()

        if key not in seen:
            seen.add(key)
            unique.append(sql)

    return unique


def extract_sql_tables(sql: str) -> List[str]:

    tables = []

    for match in SQL_TABLE_RE.finditer(sql):

        table = normalize_identifier(match.group(1))

        if table:
            tables.append(table)

    return list(dict.fromkeys(tables))


def extract_sql_aliases(sql: str) -> Dict[str, str]:

    aliases = {}

    for match in ALIAS_RE.finditer(sql):

        table = normalize_identifier(match.group(1))
        alias = normalize_identifier(match.group(2) or "")

        if alias:
            aliases[alias] = table

    return aliases


def extract_joins(sql: str) -> List[Dict[str, str]]:

    joins = []

    for match in JOIN_EQUALITY_RE.finditer(sql):

        joins.append({
            "left_alias": normalize_identifier(match.group(1)),
            "left_column": normalize_identifier(match.group(2)),
            "right_alias": normalize_identifier(match.group(3)),
            "right_column": normalize_identifier(match.group(4)),
        })

    return joins


# ============================================================
# TOP LEVEL SELECT PARSER
# ============================================================

def locate_main_select(sql: str) -> int:

    match = re.search(
        r"\bSELECT\b",
        mask_sql_comments_and_strings(sql),
        re.I
    )

    return match.start() if match else -1


def parse_select_structure(sql: str) -> Dict[str, Any]:

    sql = sql.strip()

    select_pos = locate_main_select(sql)

    if select_pos < 0:
        return {}

    masked = mask_sql_comments_and_strings(sql)

    positions = top_level_keyword_positions(
        masked[select_pos:],
        [
            "FROM",
            "WHERE",
            "GROUP BY",
            "HAVING",
            "ORDER BY",
            "UNION",
            "MINUS",
            "INTERSECT",
        ],
    )

    relative_from = positions.get("FROM")

    if relative_from is None:
        return {}

    from_pos = select_pos + relative_from

    select_text = sql[
        select_pos + len("SELECT"):
        from_pos
    ].strip()

    # Find the next top-level clause after FROM.
    clause_positions = []

    for keyword, relative_position in positions.items():

        if keyword == "FROM":
            continue

        absolute_position = select_pos + relative_position

        if absolute_position > from_pos:
            clause_positions.append(
                (absolute_position, keyword)
            )

    clause_positions.sort()

    from_end = (
        clause_positions[0][0]
        if clause_positions
        else len(sql)
    )

    from_text = sql[
        from_pos + len("FROM"):
        from_end
    ].strip()

    where_text = ""
    group_text = ""
    having_text = ""
    order_text = ""

    for index, (pos, keyword) in enumerate(clause_positions):

        end = (
            clause_positions[index + 1][0]
            if index + 1 < len(clause_positions)
            else len(sql)
        )

        body = sql[
            pos + len(keyword):
            end
        ].strip()

        if keyword == "WHERE":
            where_text = body

        elif keyword == "GROUP BY":
            group_text = body

        elif keyword == "HAVING":
            having_text = body

        elif keyword == "ORDER BY":
            order_text = body

    return {
        "select_list": split_top_level(select_text),
        "from": from_text,
        "where": where_text,
        "group_by": split_top_level(group_text),
        "having": having_text,
        "order_by": split_top_level(order_text),
        "tables": extract_sql_tables(sql),
        "aliases": extract_sql_aliases(sql),
        "joins": extract_joins(sql),
    }


# ============================================================
# SELECT OUTPUT PARSING
# ============================================================

def split_expression_alias(expression: str) -> Tuple[str, str]:

    expression = expression.strip()

    # Explicit AS alias.
    match = re.match(
        r"^(.*)\s+AS\s+([A-Za-z0-9_$#]+)$",
        expression,
        re.I | re.S
    )

    if match:
        return match.group(1).strip(), normalize_identifier(match.group(2))

    # Simple trailing alias.
    match = re.match(
        r"^(.*\S)\s+([A-Za-z][A-Za-z0-9_$#]*)$",
        expression,
        re.I | re.S
    )

    if match:

        expr = match.group(1).strip()
        alias = normalize_identifier(match.group(2))

        # Do not treat SQL keywords as aliases.
        if alias not in {
            "NULL",
            "DESC",
            "ASC",
            "END",
            "THEN",
            "ELSE",
        }:
            return expr, alias

    # Simple column.
    simple = re.match(
        r'^"?([A-Za-z0-9_$#]+)"?(?:\."?([A-Za-z0-9_$#]+)"?)?$',
        expression,
        re.I
    )

    if simple:

        if simple.group(2):
            return expression, normalize_identifier(
                simple.group(2)
            )

        return expression, normalize_identifier(
            simple.group(1)
        )

    return expression, ""


def parse_source_column(
    expression: str,
    aliases: Dict[str, str]
) -> Tuple[Optional[str], Optional[str]]:

    expression = expression.strip()

    match = re.match(
        r'^"?([A-Za-z0-9_$#]+)"?\."?([A-Za-z0-9_$#]+)"?$',
        expression
    )

    if not match:
        return None, None

    owner = normalize_identifier(match.group(1))
    column = normalize_identifier(match.group(2))

    table = aliases.get(owner, owner)

    return table, column


def detect_aggregate(expression: str) -> Optional[str]:

    match = re.match(
        r"^\s*(COUNT|SUM|AVG|MIN|MAX)\s*\(",
        expression,
        re.I
    )

    if match:
        return match.group(1).upper()

    return None


def expression_references(
    expression: str
) -> List[Tuple[str, str]]:

    return [
        (
            normalize_identifier(match.group(1)),
            normalize_identifier(match.group(2)),
        )
        for match in re.finditer(
            r'"?([A-Za-z0-9_$#]+)"?\."?([A-Za-z0-9_$#]+)"?',
            expression,
        )
    ]


def parse_outputs(
    structure: Dict[str, Any],
    metadata: Dict[str, Any]
) -> List[Dict[str, Any]]:

    outputs = []

    aliases = structure.get("aliases", {})

    for position, expression_text in enumerate(
        structure.get("select_list", [])
    ):

        expression, alias = split_expression_alias(
            expression_text
        )

        aggregate = detect_aggregate(expression)

        refs = expression_references(expression)

        source_table = None
        source_column = None

        if len(refs) == 1:

            ref_alias, ref_column = refs[0]

            source_table = aliases.get(
                ref_alias,
                ref_alias
            )

            source_column = ref_column

        elif len(refs) == 0:

            table, column = parse_source_column(
                expression,
                aliases
            )

            source_table = table
            source_column = column

        datatype = None

        if source_table and source_column:

            table_meta = metadata.get(
                "tables",
                {}
            ).get(source_table.upper(), {})

            column_meta = table_meta.get(
                "columns",
                {}
            ).get(source_column.upper(), {})

            datatype = column_meta.get("datatype")

        output_name = (
            alias
            or source_column
            or f"EXPR_{position + 1}"
        )

        is_numeric = (
            datatype in NUMERIC_TYPES
            or aggregate in {"COUNT", "SUM", "AVG"}
        )

        is_date = datatype in DATE_TYPES

        outputs.append({
            "position": position,
            "alias": output_name,
            "expression": expression,
            "source_table": source_table,
            "source_column": source_column,
            "datatype": datatype,
            "is_numeric": is_numeric,
            "is_date": is_date,
            "aggregate": aggregate,
            "references": refs,
        })

    return outputs


# ============================================================
# METADATA REFLECTION
# ============================================================

def reflect_database(connection) -> Dict[str, Any]:

    metadata = {
        "tables": {},
        "constraints": [],
        "indexes": [],
        "views": [],
        "objects": [],
    }

    print("\nReflecting Oracle metadata...")

    rows = fetch_rows(
        connection,
        """
        SELECT table_name
        FROM user_tables
        ORDER BY table_name
        """
    )

    for row in rows:

        table = normalize_identifier(row["TABLE_NAME"])

        metadata["tables"][table] = {
            "columns": {},
            "pk_columns": set(),
            "unique_columns": set(),
            "comments": "",
        }

    print(f"Tables       : {len(metadata['tables'])}")

    rows = fetch_rows(
        connection,
        """
        SELECT
            table_name,
            column_name,
            data_type,
            data_length,
            nullable,
            column_id
        FROM user_tab_columns
        ORDER BY table_name, column_id
        """
    )

    for row in rows:

        table = normalize_identifier(row["TABLE_NAME"])
        column = normalize_identifier(row["COLUMN_NAME"])

        if table not in metadata["tables"]:
            metadata["tables"][table] = {
                "columns": {},
                "pk_columns": set(),
                "unique_columns": set(),
                "comments": "",
            }

        metadata["tables"][table]["columns"][column] = {
            "datatype": normalize_identifier(
                row["DATA_TYPE"]
            ),
            "length": row["DATA_LENGTH"],
            "nullable": row["NULLABLE"],
            "column_id": row["COLUMN_ID"],
        }

    print(
        "Columns      :",
        sum(
            len(t["columns"])
            for t in metadata["tables"].values()
        )
    )

    constraints = fetch_rows(
        connection,
        """
        SELECT
            constraint_name,
            constraint_type,
            table_name,
            r_constraint_name,
            r_owner
        FROM user_constraints
        """
    )

    metadata["constraints"] = constraints

    constraint_columns = fetch_rows(
        connection,
        """
        SELECT
            constraint_name,
            table_name,
            column_name,
            position
        FROM user_cons_columns
        ORDER BY constraint_name, position
        """
    )

    constraint_map = defaultdict(list)

    for row in constraint_columns:

        constraint_map[
            normalize_identifier(row["CONSTRAINT_NAME"])
        ].append(
            normalize_identifier(row["COLUMN_NAME"])
        )

    for row in constraints:

        constraint_name = normalize_identifier(
            row["CONSTRAINT_NAME"]
        )

        constraint_type = normalize_identifier(
            row["CONSTRAINT_TYPE"]
        )

        table = normalize_identifier(
            row["TABLE_NAME"]
        )

        columns = constraint_map.get(
            constraint_name,
            []
        )

        if table not in metadata["tables"]:
            continue

        if constraint_type == "P":

            metadata["tables"][table]["pk_columns"].update(
                columns
            )

        elif constraint_type == "U":

            metadata["tables"][table]["unique_columns"].update(
                columns
            )

    print("Constraints  :", len(constraints))

    metadata["indexes"] = fetch_rows(
        connection,
        """
        SELECT
            index_name,
            table_name,
            uniqueness
        FROM user_indexes
        """
    )

    print("Indexes      :", len(metadata["indexes"]))

    metadata["views"] = fetch_rows(
        connection,
        """
        SELECT
            view_name,
            text
        FROM user_views
        """
    )

    print("Views        :", len(metadata["views"]))

    metadata["objects"] = fetch_rows(
        connection,
        """
        SELECT
            object_name,
            object_type,
            status
        FROM user_objects
        """
    )

    print("Objects      :", len(metadata["objects"]))

    # Comments.
    comments = fetch_rows(
        connection,
        """
        SELECT
            table_name,
            comments
        FROM user_tab_comments
        """
    )

    for row in comments:

        table = normalize_identifier(
            row["TABLE_NAME"]
        )

        if table in metadata["tables"]:
            metadata["tables"][table]["comments"] = (
                row["COMMENTS"] or ""
            )

    column_comments = fetch_rows(
        connection,
        """
        SELECT
            table_name,
            column_name,
            comments
        FROM user_col_comments
        """
    )

    for row in column_comments:

        table = normalize_identifier(
            row["TABLE_NAME"]
        )

        column = normalize_identifier(
            row["COLUMN_NAME"]
        )

        if (
            table in metadata["tables"]
            and column in metadata["tables"][table]["columns"]
        ):
            metadata["tables"][table]["columns"][column][
                "comments"
            ] = row["COMMENTS"] or ""

    return metadata


# ============================================================
# PACKAGE / BUSINESS SOURCE DISCOVERY
# ============================================================

def discover_static_packages(
    connection
) -> List[str]:

    rows = fetch_rows(
        connection,
        """
        SELECT object_name
        FROM user_objects
        WHERE object_type = 'PACKAGE'
          AND object_name LIKE :prefix
        ORDER BY object_name
        """,
        {"prefix": STATIC_PACKAGE_PREFIX},
    )

    return [
        normalize_identifier(row["OBJECT_NAME"])
        for row in rows[:MAX_STATIC_PACKAGES]
    ]


def read_package_body(
    connection,
    package_name: str
) -> str:

    rows = fetch_rows(
        connection,
        """
        SELECT line, text
        FROM user_source
        WHERE name = :name
          AND type = 'PACKAGE BODY'
        ORDER BY line
        """,
        {"name": package_name},
    )

    return "".join(
        row["TEXT"] or ""
        for row in rows
    )


def discover_package_members(
    source: str
) -> List[Dict[str, Any]]:

    members = []

    pattern = re.compile(
        r"\b(PROCEDURE|FUNCTION)\s+"
        r"([A-Za-z0-9_$#]+)",
        re.I
    )

    matches = list(pattern.finditer(source))

    for index, match in enumerate(matches):

        member_type = match.group(1).upper()
        member_name = normalize_identifier(match.group(2))

        start = match.start()

        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(source)
        )

        body = source[start:end]

        members.append({
            "type": member_type,
            "name": member_name,
            "source": body,
        })

        if len(members) >= MAX_STATIC_PROCEDURES:
            break

    return members


def read_static_business_layer(
    connection,
    metadata: Dict[str, Any]
) -> Dict[str, Any]:

    result = {
        "packages": [],
        "members": [],
        "queries": [],
    }

    if not ENABLE_STATIC_READING_LAYER:
        return result

    print("\nReading static business layer...")

    packages = discover_static_packages(connection)

    result["packages"] = packages

    print(f"Packages     : {len(packages)}")

    for package_name in packages:

        source = read_package_body(
            connection,
            package_name
        )

        members = discover_package_members(source)

        for member in members:

            result["members"].append({
                "package": package_name,
                "type": member["type"],
                "name": member["name"],
            })

            queries = extract_select_statements(
                member["source"]
            )

            for query in queries:

                structure = parse_select_structure(
                    query
                )

                if not structure:
                    continue

                outputs = parse_outputs(
                    structure,
                    metadata
                )

                result["queries"].append({
                    "package": package_name,
                    "member": member["name"],
                    "sql": query,
                    "tables": structure.get(
                        "tables",
                        []
                    ),
                    "aliases": structure.get(
                        "aliases",
                        {}
                    ),
                    "joins": structure.get(
                        "joins",
                        []
                    ),
                    "outputs": outputs,
                    "where": structure.get(
                        "where",
                        ""
                    ),
                    "group_by": structure.get(
                        "group_by",
                        []
                    ),
                    "having": structure.get(
                        "having",
                        ""
                    ),
                    "order_by": structure.get(
                        "order_by",
                        []
                    ),
                })

    print(
        f"Members      : {len(result['members'])}"
    )

    print(
        f"Business SQL : {len(result['queries'])}"
    )

    return result


# ============================================================
# CAPABILITY MODEL
# ============================================================

def build_capability_model(
    static_layer: Dict[str, Any],
    metadata: Dict[str, Any]
) -> List[Dict[str, Any]]:

    capabilities = []

    for query_id, query in enumerate(
        static_layer.get("queries", []),
        start=1
    ):

        outputs = query.get("outputs", [])

        dimensions = []
        measures = []
        entity_keys = []
        date_candidates = []

        for output in outputs:

            alias = normalize_identifier(
                output["alias"]
            )

            if output["is_date"]:
                date_candidates.append(output)

            if (
                output["is_numeric"]
                and not output["aggregate"]
            ):
                measures.append(output)

            if (
                output["source_table"]
                and output["source_column"]
            ):

                table = output["source_table"].upper()
                column = output["source_column"].upper()

                table_meta = metadata.get(
                    "tables",
                    {}
                ).get(table, {})

                if column in table_meta.get(
                    "pk_columns",
                    set()
                ):
                    entity_keys.append(output)

                elif column in table_meta.get(
                    "unique_columns",
                    set()
                ):
                    entity_keys.append(output)

            # A non-numeric, non-date output is naturally
            # usable as a dimension.
            if (
                not output["is_numeric"]
                and not output["is_date"]
            ):
                dimensions.append(output)

        # Existing GROUP BY expressions strengthen dimension evidence.
        group_by = query.get("group_by", [])

        for expression in group_by:

            refs = expression_references(expression)

            for _, column in refs:

                for output in outputs:

                    if (
                        output.get("source_column")
                        == column
                    ):
                        if output not in dimensions:
                            dimensions.append(output)

        # Infer a row/entity capability.
        grain_score = 0.0

        if entity_keys:
            grain_score += 0.50

        if query.get("joins"):
            grain_score += 0.20

        if outputs:
            grain_score += 0.20

        if query.get("where"):
            grain_score += 0.10

        capabilities.append({
            "capability_id": query_id,
            "package": query["package"],
            "member": query["member"],
            "sql": query["sql"],
            "tables": query["tables"],
            "aliases": query["aliases"],
            "joins": query["joins"],
            "outputs": outputs,
            "dimensions": dimensions,
            "measures": measures,
            "entity_keys": entity_keys,
            "date_candidates": date_candidates,
            "filters": query.get("where", ""),
            "group_by": query.get("group_by", []),
            "having": query.get("having", ""),
            "order_by": query.get("order_by", []),
            "grain_score": min(grain_score, 1.0),
        })

    return capabilities


# ============================================================
# REQUEST PLANNER
# ============================================================

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def extract_time_range(text: str) -> Dict[str, Any]:

    text_lower = text.lower()

    # Month + year.
    for month_name, month_number in MONTHS.items():

        pattern = rf"\b{month_name}\s+(\d{{4}})\b"

        match = re.search(
            pattern,
            text_lower
        )

        if match:

            year = int(match.group(1))

            start = f"{year:04d}-{month_number:02d}-01"

            if month_number == 12:
                next_year = year + 1
                next_month = 1
            else:
                next_year = year
                next_month = month_number + 1

            end = (
                f"{next_year:04d}-"
                f"{next_month:02d}-01"
            )

            return {
                "type": "month",
                "label": f"{month_name} {year}",
                "start": start,
                "end": end,
            }

    # ISO date.
    match = re.search(
        r"\b(\d{4}-\d{2}-\d{2})\b",
        text_lower
    )

    if match:

        date_value = match.group(1)

        return {
            "type": "date",
            "label": date_value,
            "start": date_value,
            "end": date_value,
        }

    return {}


def analyse_prompt(prompt: str) -> Dict[str, Any]:

    tokens = tokenize(prompt)

    meaningful = []

    for token in tokens:

        if token in STOPWORDS:
            continue

        meaningful.append(token)

    operation = None

    for token in tokens:

        if token in OPERATION_WORDS:

            candidate = OPERATION_WORDS[token]

            # TOTAL is deliberately not immediately treated
            # as SUM. It is resolved later from the capability.
            operation = candidate

            break

    dimension_requested = (
        any(
            token in DIMENSION_WORDS
            for token in tokens
        )
    )

    time_range = extract_time_range(prompt)

    # Detect "wise" / "by X" / "per X".
    dimension_terms = []

    wise_match = re.search(
        r"\b([a-zA-Z][a-zA-Z0-9_]*)\s+wise\b",
        prompt,
        re.I
    )

    if wise_match:
        dimension_terms.append(
            wise_match.group(1).lower()
        )

    by_match = re.search(
        r"\bby\s+([a-zA-Z][a-zA-Z0-9_]*)",
        prompt,
        re.I
    )

    if by_match:
        dimension_terms.append(
            by_match.group(1).lower()
        )

    per_match = re.search(
        r"\bper\s+([a-zA-Z][a-zA-Z0-9_]*)",
        prompt,
        re.I
    )

    if per_match:
        dimension_terms.append(
            per_match.group(1).lower()
        )

    # Remove grammatical request words from concept terms.
    concepts = [
        token
        for token in meaningful
        if token not in {
            "wise",
            "group",
            "grouped",
            "breakdown",
            "break",
            "month",
            "year",
            "quarter",
            "week",
            "date",
            "between",
        }
    ]

    # If an explicit dimension phrase exists, remove that word
    # from subject terms.
    for dimension_term in dimension_terms:
        concepts = [
            x for x in concepts
            if x != dimension_term
        ]

    # The key correction:
    #
    # "total policies"
    # does NOT become SUM.
    #
    # TOTAL is left unresolved until capability matching.
    #
    # "total contribution"
    # can become SUM only when contribution is found
    # as a numeric capability.

    return {
        "prompt": prompt,
        "tokens": tokens,
        "concept_terms": concepts,
        "subject_terms": concepts,
        "measure_terms": concepts,
        "dimension_terms": dimension_terms,
        "dimension_requested": dimension_requested,
        "operation_requested": operation,
        "time_range": time_range,
    }


# ============================================================
# CAPABILITY TEXT
# ============================================================

def capability_text(
    capability: Dict[str, Any]
) -> str:

    parts = [
        capability.get("package", ""),
        capability.get("member", ""),
    ]

    for output in capability.get("outputs", []):

        parts.extend([
            output.get("alias", ""),
            output.get("source_column", "") or "",
            output.get("expression", ""),
        ])

    for table in capability.get("tables", []):
        parts.append(table)

    return " ".join(parts).lower()


def score_term_against_output(
    term: str,
    output: Dict[str, Any]
) -> float:

    candidates = [
        output.get("alias", ""),
        output.get("source_column", "") or "",
        output.get("source_table", "") or "",
        output.get("expression", ""),
    ]

    best = 0.0

    term_forms = word_forms(term)

    for candidate in candidates:

        candidate_parts = set(
            identifier_parts(candidate)
        )

        candidate_text = candidate.lower()

        for form in term_forms:

            if form in candidate_parts:
                best = max(best, 1.0)

            elif form in candidate_text:
                best = max(best, 0.85)

            else:
                for part in candidate_parts:
                    best = max(
                        best,
                        similarity(form, part)
                    )

    return min(best, 1.0)


# ============================================================
# CAPABILITY MATCHING
# ============================================================

def score_capability(
    capability: Dict[str, Any],
    request: Dict[str, Any]
) -> Dict[str, Any]:

    subject_terms = request.get(
        "subject_terms",
        []
    )

    dimension_terms = request.get(
        "dimension_terms",
        []
    )

    measure_terms = request.get(
        "measure_terms",
        []
    )

    subject_score = 0.0
    measure_score = 0.0
    dimension_score = 0.0
    time_score = 0.0
    entity_score = 0.0

    # --------------------------------------------------------
    # Subject
    # --------------------------------------------------------

    for term in subject_terms:

        local = 0.0

        for output in capability.get(
            "outputs",
            []
        ):

            local = max(
                local,
                score_term_against_output(
                    term,
                    output
                )
            )

        subject_score = max(
            subject_score,
            local
        )

    # --------------------------------------------------------
    # Measure
    # --------------------------------------------------------

    for term in measure_terms:

        for output in capability.get(
            "measures",
            []
        ):

            measure_score = max(
                measure_score,
                score_term_against_output(
                    term,
                    output
                )
            )

        # Numeric outputs are valid candidate measures even
        # when the language is not an exact identifier.
        for output in capability.get(
            "outputs",
            []
        ):

            if output.get("is_numeric"):
                measure_score = max(
                    measure_score,
                    score_term_against_output(
                        term,
                        output
                    ) * 0.85
                )

    # --------------------------------------------------------
    # Dimension
    # --------------------------------------------------------

    for term in dimension_terms:

        for output in capability.get(
            "dimensions",
            []
        ):

            dimension_score = max(
                dimension_score,
                score_term_against_output(
                    term,
                    output
                )
            )

    # --------------------------------------------------------
    # Date capability
    # --------------------------------------------------------

    if request.get("time_range"):

        if capability.get("date_candidates"):

            time_score = 1.0

        elif re.search(
            r"\bDATE\b|\bTIMESTAMP\b|_DATE\b|DATE_",
            capability_text(capability),
            re.I
        ):
            time_score = 0.60

    # --------------------------------------------------------
    # Entity capability
    # --------------------------------------------------------

    if capability.get("entity_keys"):

        entity_score = 1.0

    elif capability.get("outputs"):

        entity_score = 0.40

    # --------------------------------------------------------
    # Query completeness
    # --------------------------------------------------------

    completeness = 0.0

    if subject_score > 0:
        completeness += 0.30

    if request.get("dimension_requested"):

        if dimension_score > 0:
            completeness += 0.25

    else:
        completeness += 0.20

    if request.get("time_range"):

        if time_score > 0:
            completeness += 0.20

    else:
        completeness += 0.15

    if entity_score > 0:
        completeness += 0.10

    if capability.get("joins"):
        completeness += 0.05

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    score = (
        subject_score * 0.30
        + measure_score * 0.25
        + dimension_score * 0.20
        + time_score * 0.10
        + entity_score * 0.05
        + completeness * 0.10
    )

    return {
        "capability": capability,
        "score": round(score, 4),
        "subject_score": round(subject_score, 4),
        "measure_score": round(measure_score, 4),
        "dimension_score": round(dimension_score, 4),
        "time_score": round(time_score, 4),
        "entity_score": round(entity_score, 4),
        "completeness": round(completeness, 4),
    }


def match_capabilities(
    capabilities: List[Dict[str, Any]],
    request: Dict[str, Any]
) -> List[Dict[str, Any]]:

    scored = [
        score_capability(
            capability,
            request
        )
        for capability in capabilities
    ]

    scored.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return scored[:MAX_CANDIDATES]


# ============================================================
# OUTPUT RESOLUTION
# ============================================================

def resolve_output(
    capability: Dict[str, Any],
    terms: List[str],
    mode: str
) -> Optional[Dict[str, Any]]:

    candidates = []

    if mode == "dimension":
        candidates = capability.get(
            "dimensions",
            []
        )

    elif mode == "measure":
        candidates = [
            x for x in capability.get(
                "outputs",
                []
            )
            if x.get("is_numeric")
        ]

    elif mode == "date":
        candidates = capability.get(
            "date_candidates",
            []
        )

    elif mode == "entity":
        candidates = capability.get(
            "entity_keys",
            []
        )

    if not candidates:
        candidates = capability.get(
            "outputs",
            []
        )

    best = None
    best_score = 0.0

    for candidate in candidates:

        local = 0.0

        for term in terms:

            local = max(
                local,
                score_term_against_output(
                    term,
                    candidate
                )
            )

        if local > best_score:

            best_score = local
            best = candidate

    return best


def resolve_query_plan(
    match: Dict[str, Any],
    request: Dict[str, Any]
) -> Dict[str, Any]:

    capability = match["capability"]

    subject = resolve_output(
        capability,
        request.get(
            "subject_terms",
            []
        ),
        "entity"
    )

    measure = resolve_output(
        capability,
        request.get(
            "measure_terms",
            []
        ),
        "measure"
    )

    dimension = resolve_output(
        capability,
        request.get(
            "dimension_terms",
            []
        ),
        "dimension"
    )

    date_output = resolve_output(
        capability,
        [],
        "date"
    )

    operation_requested = request.get(
        "operation_requested"
    )

    # --------------------------------------------------------
    # Generic operation resolution
    # --------------------------------------------------------

    if operation_requested == "TOTAL":

        if measure:

            # A numeric measure means TOTAL can be SUM.
            operation = "SUM"

        elif subject:

            # An entity/key means TOTAL means count.
            operation = "COUNT"

        else:

            operation = "COUNT"

    elif operation_requested == "COUNT":

        operation = "COUNT"

    elif operation_requested in {
        "SUM",
        "AVG",
        "MIN",
        "MAX",
    }:

        operation = operation_requested

    else:

        if measure:
            operation = "SUM"
        elif subject:
            operation = "COUNT"
        else:
            operation = "SELECT"

    # --------------------------------------------------------
    # Entity key
    # --------------------------------------------------------

    entity_key = None

    if capability.get("entity_keys"):

        entity_key = capability["entity_keys"][0]

    if not entity_key and subject:

        entity_key = subject

    return {
        "intent": (
            "aggregate"
            if operation != "SELECT"
            else "select"
        ),
        "operation": operation,
        "subject": subject,
        "measure": measure,
        "dimension": dimension,
        "entity_key": entity_key,
        "date": date_output,
        "time_range": request.get(
            "time_range"
        ),
        "capability": capability,
        "match_score": match["score"],
    }


# ============================================================
# SQL IDENTIFIER SAFETY
# ============================================================

def quote_identifier(identifier: str) -> str:

    identifier = normalize_identifier(identifier)

    if not re.match(
        r"^[A-Z0-9_$#]+$",
        identifier
    ):
        raise ValueError(
            f"Unsafe SQL identifier: {identifier}"
        )

    return f'"{identifier}"'


# ============================================================
# STATIC QUERY COMPOSITION
# ============================================================

def compose_query(
    plan: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:

    capability = plan["capability"]

    base_sql = capability["sql"].strip()

    # Remove trailing semicolon.
    base_sql = base_sql.rstrip(";").strip()

    operation = plan["operation"]

    entity_key = plan.get("entity_key")
    measure = plan.get("measure")
    dimension = plan.get("dimension")
    date_output = plan.get("date")
    time_range = plan.get("time_range")

    params = {}

    # --------------------------------------------------------
    # Build inner aliases.
    # --------------------------------------------------------

    if entity_key:

        entity_alias = quote_identifier(
            entity_key["alias"]
        )

    else:
        entity_alias = None

    if measure:

        measure_alias = quote_identifier(
            measure["alias"]
        )

    else:
        measure_alias = None

    if dimension:

        dimension_alias = quote_identifier(
            dimension["alias"]
        )

    else:
        dimension_alias = None

    # --------------------------------------------------------
    # COUNT
    # --------------------------------------------------------

    if operation == "COUNT":

        if entity_alias:

            select_expression = (
                f"COUNT(DISTINCT {entity_alias})"
            )

        else:

            select_expression = "COUNT(*)"

        sql = f"""
SELECT
    {select_expression} AS RESULT_COUNT
FROM (
    {base_sql}
) Q
"""

    # --------------------------------------------------------
    # SUM
    # --------------------------------------------------------

    elif operation == "SUM":

        if not measure_alias:

            raise ValueError(
                "SUM requested but no numeric measure "
                "was resolved from the business capability."
            )

        sql = f"""
SELECT
    SUM({measure_alias}) AS RESULT_TOTAL
FROM (
    {base_sql}
) Q
"""

    # --------------------------------------------------------
    # AVG
    # --------------------------------------------------------

    elif operation == "AVG":

        if not measure_alias:

            raise ValueError(
                "AVG requested but no numeric measure "
                "was resolved."
            )

        sql = f"""
SELECT
    AVG({measure_alias}) AS RESULT_AVERAGE
FROM (
    {base_sql}
) Q
"""

    # --------------------------------------------------------
    # MIN
    # --------------------------------------------------------

    elif operation == "MIN":

        if not measure_alias:

            raise ValueError(
                "MIN requested but no measure was resolved."
            )

        sql = f"""
SELECT
    MIN({measure_alias}) AS RESULT_MIN
FROM (
    {base_sql}
) Q
"""

    # --------------------------------------------------------
    # MAX
    # --------------------------------------------------------

    elif operation == "MAX":

        if not measure_alias:

            raise ValueError(
                "MAX requested but no measure was resolved."
            )

        sql = f"""
SELECT
    MAX({measure_alias}) AS RESULT_MAX
FROM (
    {base_sql}
) Q
"""

    else:

        sql = f"""
SELECT *
FROM (
    {base_sql}
) Q
"""

    # --------------------------------------------------------
    # DIMENSION
    # --------------------------------------------------------

    if dimension_alias:

        if operation == "COUNT":

            if entity_alias:

                aggregation = (
                    f"COUNT(DISTINCT {entity_alias})"
                )

            else:

                aggregation = "COUNT(*)"

            sql = f"""
SELECT
    {dimension_alias} AS DIMENSION_VALUE,
    {aggregation} AS RESULT_COUNT
FROM (
    {base_sql}
) Q
"""

        elif operation == "SUM":

            sql = f"""
SELECT
    {dimension_alias} AS DIMENSION_VALUE,
    SUM({measure_alias}) AS RESULT_TOTAL
FROM (
    {base_sql}
) Q
"""

        elif operation == "AVG":

            sql = f"""
SELECT
    {dimension_alias} AS DIMENSION_VALUE,
    AVG({measure_alias}) AS RESULT_AVERAGE
FROM (
    {base_sql}
) Q
"""

        elif operation == "MIN":

            sql = f"""
SELECT
    {dimension_alias} AS DIMENSION_VALUE,
    MIN({measure_alias}) AS RESULT_MIN
FROM (
    {base_sql}
) Q
"""

        elif operation == "MAX":

            sql = f"""
SELECT
    {dimension_alias} AS DIMENSION_VALUE,
    MAX({measure_alias}) AS RESULT_MAX
FROM (
    {base_sql}
) Q
"""

        sql += f"""
GROUP BY
    {dimension_alias}
ORDER BY
    {dimension_alias}
"""

    # --------------------------------------------------------
    # DATE FILTER
    # --------------------------------------------------------

    if time_range and date_output:
        date_col = date_output["source_column"] or date_output["alias"]
        
        # Inject the date criteria directly into the base SQL query block safely
        where_cond = f" {date_col} >= DATE '{time_range['start']}' AND {date_col} < DATE '{time_range['end']}' "
        
        if "WHERE" in base_sql.upper():
            # Append if a WHERE clause already exists in the package layout
            base_sql = re.sub(r"(\bWHERE\b)", r"\1" + where_cond + " AND ", base_sql, count=1, flags=re.I)
        else:
            # Construct a new WHERE clause right before GROUP BY or ORDER BY if missing
            if "GROUP BY" in base_sql.upper():
                base_sql = re.sub(r"(\bGROUP BY\b)", r"WHERE " + where_cond + r" \1", base_sql, count=1, flags=re.I)
            elif "ORDER BY" in base_sql.upper():
                base_sql = re.sub(r"(\bORDER BY\b)", r"WHERE " + where_cond + r" \1", base_sql, count=1, flags=re.I)
            else:
                base_sql = base_sql + " WHERE " + where_cond

        # Re-build the wrapping query layout using the updated base_sql text layout
        # (This preserves your exact outer block logic mapping without changing structure)
        if dimension_alias:
            if operation == "COUNT":
                aggregation = f"COUNT(DISTINCT {entity_alias})" if entity_alias else "COUNT(*)"
                sql = f"SELECT {dimension_alias} AS DIMENSION_VALUE, {aggregation} AS RESULT_COUNT FROM ({base_sql}) Q GROUP BY {dimension_alias} ORDER BY {dimension_alias}"
            elif operation == "SUM":
                sql = f"SELECT {dimension_alias} AS DIMENSION_VALUE, SUM({measure_alias}) AS RESULT_TOTAL FROM ({base_sql}) Q GROUP BY {dimension_alias} ORDER BY {dimension_alias}"



# ============================================================
# EXPLAIN PLAN VALIDATION
# ============================================================

def validate_sql(
    connection,
    sql: str,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:

    cursor = connection.cursor()

    try:

        cursor.execute(
            "EXPLAIN PLAN FOR " + sql,
            params or {}
        )

        rows = fetch_rows(
            connection,
            """
            SELECT
                plan_table_output
            FROM TABLE(
                DBMS_XPLAN.DISPLAY(
                    'PLAN_TABLE',
                    NULL,
                    'BASIC'
                )
            )
            """
        )

        plan_text = "\n".join(
            str(row.get("PLAN_TABLE_OUTPUT", ""))
            for row in rows
        )

        return {
            "valid": True,
            "plan": plan_text,
        }

    except Exception as exc:

        return {
            "valid": False,
            "error": str(exc),
            "plan": "",
        }

    finally:
        cursor.close()


# ============================================================
# EXECUTION
# ============================================================

def run_query(
    connection,
    sql: str,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:

    rows, columns = execute_sql(
        connection,
        sql,
        params
    )

    return {
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
    }


# ============================================================
# DISPLAY
# ============================================================

def print_request(request: Dict[str, Any]):

    print("\n" + "=" * 70)
    print("REQUEST PLAN")
    print("=" * 70)

    print(
        "Prompt             :",
        request["prompt"]
    )

    print(
        "Concepts           :",
        request["concept_terms"]
    )

    print(
        "Dimension terms    :",
        request["dimension_terms"]
    )

    print(
        "Operation requested:",
        request["operation_requested"]
    )

    print(
        "Time range         :",
        request["time_range"]
    )


def print_capability_matches(
    matches: List[Dict[str, Any]]
):

    print("\n" + "=" * 70)
    print("CAPABILITY MATCHES")
    print("=" * 70)

    for index, match in enumerate(
        matches[:10],
        start=1
    ):

        capability = match["capability"]

        print(
            f"\n[{index}] "
            f"{capability['package']}."
            f"{capability['member']}"
        )

        print(
            f"    Score      : {match['score']}"
        )

        print(
            f"    Subject    : {match['subject_score']}"
        )

        print(
            f"    Measure    : {match['measure_score']}"
        )

        print(
            f"    Dimension  : {match['dimension_score']}"
        )

        print(
            f"    Time       : {match['time_score']}"
        )

        print(
            f"    Entity     : {match['entity_score']}"
        )

        print(
            "    Outputs    :",
            [
                x["alias"]
                for x in capability.get(
                    "outputs",
                    []
                )
            ]
        )


def print_plan(
    plan: Dict[str, Any]
):

    print("\n" + "=" * 70)
    print("RESOLVED QUERY PLAN")
    print("=" * 70)

    print(
        "Intent       :",
        plan["intent"]
    )

    print(
        "Operation    :",
        plan["operation"]
    )

    print(
        "Subject      :",
        (
            plan["subject"]["alias"]
            if plan["subject"]
            else None
        )
    )

    print(
        "Entity Key   :",
        (
            plan["entity_key"]["alias"]
            if plan["entity_key"]
            else None
        )
    )

    print(
        "Measure      :",
        (
            plan["measure"]["alias"]
            if plan["measure"]
            else None
        )
    )

    print(
        "Dimension    :",
        (
            plan["dimension"]["alias"]
            if plan["dimension"]
            else None
        )
    )

    print(
        "Date         :",
        (
            plan["date"]["alias"]
            if plan["date"]
            else None
        )
    )

    print(
        "Capability   :",
        (
            f"{plan['capability']['package']}."
            f"{plan['capability']['member']}"
        )
    )

    print(
        "Match score  :",
        plan["match_score"]
    )


def print_results(
    result: Dict[str, Any]
):

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    columns = result["columns"]

    print("Columns:", columns)

    for row in result["rows"][:100]:

        print(row)

    if result["row_count"] > 100:

        print(
            f"... {result['row_count'] - 100} more rows"
        )


# ============================================================
# COMPLETE ENGINE
# ============================================================

def process_prompt(
    connection,
    prompt: str,
    metadata: Dict[str, Any],
    capabilities: List[Dict[str, Any]]
):

    request = analyse_prompt(prompt)

    print_request(request)

    if not capabilities:

        raise RuntimeError(
            "No static business SQL capabilities were discovered."
        )

    matches = match_capabilities(
        capabilities,
        request
    )

    print_capability_matches(matches)

    if not matches:

        raise RuntimeError(
            "No business capability matched the request."
        )

    best = matches[0]

    if best["score"] < MIN_CAPABILITY_SCORE:

        raise RuntimeError(
            "Business meaning could not be resolved "
            "with sufficient evidence."
        )

    plan = resolve_query_plan(
        best,
        request
    )

    print_plan(plan)

    sql, params = compose_query(plan)

    print("\n" + "=" * 70)
    print("GENERATED SQL")
    print("=" * 70)

    print(sql)

    print("\n" + "=" * 70)
    print("ORACLE VALIDATION")
    print("=" * 70)

    validation = validate_sql(
        connection,
        sql,
        params
    )

    if not validation["valid"]:

        print("VALIDATION FAILED")
        print(validation["error"])

        return {
            "request": request,
            "plan": plan,
            "sql": sql,
            "validation": validation,
            "result": None,
        }

    print("SQL VALID")

    print("\n" + validation["plan"])

    # Execute only after successful explain-plan validation.
    try:

        result = run_query(
            connection,
            sql,
            params
        )

        print_results(result)

    except Exception as exc:

        print("\nEXECUTION FAILED")
        print(str(exc))

        result = {
            "error": str(exc)
        }

    return {
        "request": request,
        "plan": plan,
        "sql": sql,
        "validation": validation,
        "result": result,
    }


# ============================================================
# CAPABILITY REPORT
# ============================================================

def print_capability_report(
    capabilities: List[Dict[str, Any]]
):

    print("\n" + "=" * 70)
    print("STATIC BUSINESS CAPABILITY MODEL")
    print("=" * 70)

    for capability in capabilities:

        print(
            f"\nCapability #{capability['capability_id']}"
        )

        print(
            "Package :",
            capability["package"]
        )

        print(
            "Member  :",
            capability["member"]
        )

        print(
            "Tables  :",
            ", ".join(
                capability["tables"]
            )
        )

        print(
            "Keys    :",
            [
                x["alias"]
                for x in capability.get(
                    "entity_keys",
                    []
                )
            ]
        )

        print(
            "Measures:",
            [
                x["alias"]
                for x in capability.get(
                    "measures",
                    []
                )
            ]
        )

        print(
            "Dimensions:",
            [
                x["alias"]
                for x in capability.get(
                    "dimensions",
                    []
                )
            ]
        )


# ============================================================
# LAB
# ============================================================

def run_lab():

    print("=" * 70)
    print("DYNAMIC ORACLE BUSINESS INTELLIGENCE ENGINE")
    print("=" * 70)

    print(
        f"Database: {DATABASE_NAME}"
    )

    connection = None

    try:

        print("\nConnecting to Oracle...")

        connection = get_db_connection(
            DATABASE_NAME
        )

        print("Connected.")

        # ----------------------------------------------------
        # 1. Metadata
        # ----------------------------------------------------

        metadata = reflect_database(
            connection
        )

        # ----------------------------------------------------
        # 2. Static business reading
        # ----------------------------------------------------

        static_layer = read_static_business_layer(
            connection,
            metadata
        )

        # ----------------------------------------------------
        # 3. Capability model
        # ----------------------------------------------------

        capabilities = build_capability_model(
            static_layer,
            metadata
        )

        print(
            "\nCapabilities  :",
            len(capabilities)
        )

        print_capability_report(
            capabilities
        )

        # ----------------------------------------------------
        # 4. Test prompts
        # ----------------------------------------------------

        prompts = [
            "Let me have the total policies",

            "How many policies are there",

            "Let me know the business for the month August 2026",

            "Give me region wise business for August 2026",

            "Show policy status wise contribution for August 2026",
        ]

        for prompt in prompts:

            print("\n\n")
            print("#" * 70)
            print(
                "PROMPT:",
                prompt
            )
            print("#" * 70)

            try:

                process_prompt(
                    connection,
                    prompt,
                    metadata,
                    capabilities
                )

            except Exception as exc:

                print(
                    "\nENGINE ERROR:",
                    str(exc)
                )

    finally:

        if connection:

            connection.close()

        print("\nConnection closed.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_lab()