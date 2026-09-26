# dynamic_engine.py
#
# Generic Oracle Business Intelligence / NL-to-SQL Engine
#
# Architecture:
#
#   Natural Language
#          ↓
#   Generic Request Planner
#          ↓
#   Proven Business SQL Evidence
#          ↓
#   Capability Matching
#          ↓
#   Query Plan Resolution
#          ↓
#   Query Composition
#          ↓
#   Oracle Validation
#          ↓
#   Result
#
# IMPORTANT:
#   - databases.py is NOT modified.
#   - No business vocabulary mapping is hardcoded.
#   - No prompt-specific business rules are used.
#   - Proven SQL from business packages is the primary semantic evidence.
#   - Entity/key evidence is kept separate from numeric measure evidence.
#   - Date evidence is derived from proven SQL and Oracle metadata.
#   - The engine starts once and then accepts interactive prompts.
#

import re
import difflib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from databases import get_db_connection


# ============================================================
# CONFIGURATION
# ============================================================

DATABASE_NAME = "PQTEPF"

MAX_SOURCE_OBJECTS = 40
MAX_HISTORY_ROWS = 500
MAX_CANDIDATES = 12

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

    return re.findall(
        r"[a-z0-9_$#]+",
        text
    )


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

    return difflib.SequenceMatcher(
        None,
        a,
        b
    ).ratio()


def identifier_parts(identifier: str) -> List[str]:

    identifier = normalize_identifier(identifier)

    identifier = identifier.replace('"', "")

    identifier = re.sub(
        r"([a-z])([A-Z])",
        r"\1_\2",
        identifier
    )

    return [
        p.lower()
        for p in re.split(
            r"[_.$#\s]+",
            identifier
        )
        if p
    ]


def compact_identifier(identifier: str) -> str:

    return "".join(
        identifier_parts(identifier)
    )


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

        cursor.execute(
            sql,
            params or {}
        )

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

        cursor.execute(
            sql,
            params or {}
        )

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


COLUMN_REFERENCE_RE = re.compile(
    r"""
    "?([A-Za-z0-9_$#]+)"?
    \.
    "?([A-Za-z0-9_$#]+)"?
    """,
    re.I | re.X,
)


def mask_sql_comments_and_strings(
    sql: str
) -> str:
    """
    Preserve SQL positions while masking comments
    and string literals.
    """

    result = list(sql)

    i = 0
    n = len(sql)

    while i < n:

        # ----------------------------------------------------
        # Single-line comment
        # ----------------------------------------------------

        if i + 1 < n and sql[i:i + 2] == "--":

            j = i

            while j < n and sql[j] != "\n":

                result[j] = " "

                j += 1

            i = j

            continue

        # ----------------------------------------------------
        # Block comment
        # ----------------------------------------------------

        if i + 1 < n and sql[i:i + 2] == "/*":

            j = i + 2

            while (
                j + 1 < n
                and sql[j:j + 2] != "*/"
            ):

                result[j] = " "

                j += 1

            if j + 1 < n:

                result[j] = " "
                result[j + 1] = " "

                j += 2

            i = j

            continue

        # ----------------------------------------------------
        # String literal
        # ----------------------------------------------------

        if sql[i] == "'":

            result[i] = " "

            j = i + 1

            while j < n:

                result[j] = " "

                if sql[j] == "'":

                    if (
                        j + 1 < n
                        and sql[j + 1] == "'"
                    ):

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

            depth = max(
                0,
                depth - 1
            )

            i += 1

            continue

        if depth == 0:

            for keyword in keywords:

                if upper.startswith(
                    keyword,
                    i
                ):

                    before_ok = (
                        i == 0
                        or not upper[i - 1].isalnum()
                    )

                    after_pos = (
                        i + len(keyword)
                    )

                    after_ok = (
                        after_pos >= len(upper)
                        or not upper[
                            after_pos
                        ].isalnum()
                    )

                    if (
                        before_ok
                        and after_ok
                    ):

                        positions.setdefault(
                            keyword,
                            i
                        )

        i += 1

    return positions


def split_top_level(
    text: str,
    delimiter: str = ","
) -> List[str]:

    result = []

    start = 0
    depth = 0

    masked = mask_sql_comments_and_strings(
        text
    )

    for i, ch in enumerate(masked):

        if ch == "(":

            depth += 1

        elif ch == ")":

            depth = max(
                0,
                depth - 1
            )

        elif (
            ch == delimiter
            and depth == 0
        ):

            part = text[
                start:i
            ].strip()

            if part:

                result.append(part)

            start = i + 1

    final = text[
        start:
    ].strip()

    if final:

        result.append(final)

    return result


def extract_select_statements(
    source: str
) -> List[str]:
    """
    Extract executable static SELECT statements from PL/SQL source.

    A SELECT containing nested SELECTs is treated as ONE business SQL
    statement. Nested SELECTs remain inside the parent SQL.

    Important:
    PL/SQL cursor FOR queries may appear as:

        FOR R IN (
            SELECT ...
        )

    Therefore a SELECT at parenthesis depth > 0 is not automatically
    considered a nested subquery.

    Dynamic EXECUTE IMMEDIATE SQL is not guessed.
    """

    if not source:
        return []

    masked = mask_sql_comments_and_strings(
        source
    )

    statements = []

    select_matches = list(
        re.finditer(
            r"\bSELECT\b",
            masked,
            re.I
        )
    )

    if not select_matches:
        return []

    # ------------------------------------------------------------
    # Calculate parenthesis depth at every SELECT.
    # ------------------------------------------------------------

    depth = 0
    depth_at_select = {}

    select_positions = {
        match.start()
        for match in select_matches
    }

    for position, character in enumerate(masked):

        if character == "(":
            depth += 1

        elif character == ")":
            depth = max(
                0,
                depth - 1
            )

        if position in select_positions:
            depth_at_select[position] = depth

    # ------------------------------------------------------------
    # Determine whether a SELECT inside parentheses is actually
    # a PL/SQL cursor query rather than a nested SQL subquery.
    # ------------------------------------------------------------

    def is_cursor_select(
        select_position: int
    ) -> bool:

        prefix = masked[
            max(
                0,
                select_position - 500
            ):
            select_position
        ]

        # Cursor FOR loop:
        #
        # FOR rec IN (
        #     SELECT ...
        # )
        #
        cursor_for = re.search(
            r"\bIN\s*\(\s*$",
            prefix,
            re.I | re.S
        )

        if cursor_for:
            return True

        # OPEN cursor FOR (
        #     SELECT ...
        # )
        #
        open_for = re.search(
            r"\bFOR\s*\(\s*$",
            prefix,
            re.I | re.S
        )

        if open_for:
            return True

        return False

    # ------------------------------------------------------------
    # Identify root executable SELECT statements.
    # ------------------------------------------------------------

    root_select_positions = []

    for match in select_matches:

        start = match.start()

        depth = depth_at_select.get(
            start,
            0
        )

        # Normal SQL:
        #
        # SELECT ...
        #
        if depth == 0:

            root_select_positions.append(
                start
            )

            continue

        # PL/SQL cursor FOR query:
        #
        # FOR R IN (
        #     SELECT ...
        # )
        #
        if is_cursor_select(start):

            root_select_positions.append(
                start
            )

    # ------------------------------------------------------------
    # Safety fallback.
    #
    # If PL/SQL structure prevented recognition, use the first
    # SELECT as the business SQL root rather than returning zero.
    # ------------------------------------------------------------

    if not root_select_positions:

        root_select_positions = [
            select_matches[0].start()
        ]

    # ------------------------------------------------------------
    # Extract each root SQL statement.
    #
    # We terminate only at a semicolon at parenthesis depth zero
    # relative to the SELECT.
    # ------------------------------------------------------------

    for start in root_select_positions:

        local_depth = 0
        end = None

        for position in range(
            start,
            len(masked)
        ):

            character = masked[position]

            if character == "(":

                local_depth += 1

            elif character == ")":

                local_depth = max(
                    0,
                    local_depth - 1
                )

            elif (
                character == ";"
                and local_depth == 0
            ):

                end = position
                break

        if end is None:

            sql = source[
                start:
            ].strip()

        else:

            sql = source[
                start:end
            ].strip()

        if not sql:
            continue

        if not re.match(
            r"^\s*SELECT\b",
            sql,
            re.I
        ):
            continue

        if len(sql) > MAX_STATIC_QUERY_CHARS:
            continue

        statements.append(
            sql
        )

    # ------------------------------------------------------------
    # Deduplicate.
    # ------------------------------------------------------------

    unique = []
    seen = set()

    for sql in statements:

        key = normalize_space(
            sql
        ).upper()

        if key in seen:
            continue

        seen.add(key)
        unique.append(sql)

    return unique


def extract_sql_tables(
    sql: str
) -> List[str]:

    tables = []

    for match in SQL_TABLE_RE.finditer(sql):

        table = normalize_identifier(
            match.group(1)
        )

        if table:

            tables.append(table)

    return list(
        dict.fromkeys(tables)
    )


def extract_sql_aliases(
    sql: str
) -> Dict[str, str]:

    aliases = {}

    for match in ALIAS_RE.finditer(sql):

        table = normalize_identifier(
            match.group(1)
        )

        alias = normalize_identifier(
            match.group(2) or ""
        )

        if alias:

            aliases[alias] = table

    return aliases


def extract_joins(
    sql: str
) -> List[Dict[str, str]]:

    joins = []

    for match in JOIN_EQUALITY_RE.finditer(
        sql
    ):

        joins.append({
            "left_alias": normalize_identifier(
                match.group(1)
            ),
            "left_column": normalize_identifier(
                match.group(2)
            ),
            "right_alias": normalize_identifier(
                match.group(3)
            ),
            "right_column": normalize_identifier(
                match.group(4)
            ),
        })

    return joins


def extract_column_references(
    sql: str
) -> List[Tuple[str, str]]:

    references = []

    masked = mask_sql_comments_and_strings(
        sql
    )

    for match in COLUMN_REFERENCE_RE.finditer(
        masked
    ):

        references.append((
            normalize_identifier(
                match.group(1)
            ),
            normalize_identifier(
                match.group(2)
            ),
        ))

    return list(
        dict.fromkeys(references)
    )


# ============================================================
# SELECT STRUCTURE
# ============================================================

def locate_main_select(
    sql: str
) -> int:

    match = re.search(
        r"\bSELECT\b",
        mask_sql_comments_and_strings(sql),
        re.I
    )

    return (
        match.start()
        if match
        else -1
    )


def parse_select_structure(
    sql: str
) -> Dict[str, Any]:

    sql = sql.strip()

    select_pos = locate_main_select(
        sql
    )

    if select_pos < 0:

        return {}

    masked = mask_sql_comments_and_strings(
        sql
    )

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

    relative_from = positions.get(
        "FROM"
    )

    if relative_from is None:

        return {}

    from_pos = (
        select_pos
        + relative_from
    )

    select_text = sql[
        select_pos
        + len("SELECT"):
        from_pos
    ].strip()

    clause_positions = []

    for keyword, relative_position in (
        positions.items()
    ):

        if keyword == "FROM":

            continue

        absolute_position = (
            select_pos
            + relative_position
        )

        if absolute_position > from_pos:

            clause_positions.append((
                absolute_position,
                keyword
            ))

    clause_positions.sort()

    from_end = (
        clause_positions[0][0]
        if clause_positions
        else len(sql)
    )

    from_text = sql[
        from_pos
        + len("FROM"):
        from_end
    ].strip()

    where_text = ""
    group_text = ""
    having_text = ""
    order_text = ""

    for index, (
        pos,
        keyword
    ) in enumerate(
        clause_positions
    ):

        end = (
            clause_positions[index + 1][0]
            if index + 1
            < len(clause_positions)
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
        "select_list": split_top_level(
            select_text
        ),
        "from": from_text,
        "where": where_text,
        "group_by": split_top_level(
            group_text
        ),
        "having": having_text,
        "order_by": split_top_level(
            order_text
        ),
        "tables": extract_sql_tables(
            sql
        ),
        "aliases": extract_sql_aliases(
            sql
        ),
        "joins": extract_joins(
            sql
        ),
        "column_references":
            extract_column_references(sql),
    }


# ============================================================
# SELECT OUTPUT PARSING
# ============================================================

def split_expression_alias(
    expression: str
) -> Tuple[str, str]:

    expression = expression.strip()

    # --------------------------------------------------------
    # Explicit AS.
    # --------------------------------------------------------

    match = re.match(
        r"^(.*)\s+AS\s+([A-Za-z0-9_$#]+)$",
        expression,
        re.I | re.S
    )

    if match:

        return (
            match.group(1).strip(),
            normalize_identifier(
                match.group(2)
            )
        )

    # --------------------------------------------------------
    # Simple trailing alias.
    # --------------------------------------------------------

    match = re.match(
        r"^(.*\S)\s+([A-Za-z][A-Za-z0-9_$#]*)$",
        expression,
        re.I | re.S
    )

    if match:

        expr = match.group(
            1
        ).strip()

        alias = normalize_identifier(
            match.group(2)
        )

        if alias not in {
            "NULL",
            "DESC",
            "ASC",
            "END",
            "THEN",
            "ELSE",
        }:

            return (
                expr,
                alias
            )

    # --------------------------------------------------------
    # Simple column.
    # --------------------------------------------------------

    simple = re.match(
        r'^"?([A-Za-z0-9_$#]+)"?'
        r'(?:\."?([A-Za-z0-9_$#]+)"?)?$',
        expression,
        re.I
    )

    if simple:

        if simple.group(2):

            return (
                expression,
                normalize_identifier(
                    simple.group(2)
                )
            )

        return (
            expression,
            normalize_identifier(
                simple.group(1)
            )
        )

    return expression, ""


def parse_source_column(
    expression: str,
    aliases: Dict[str, str]
) -> Tuple[
    Optional[str],
    Optional[str]
]:

    expression = expression.strip()

    match = re.match(
        r'^"?([A-Za-z0-9_$#]+)"?'
        r'\."?([A-Za-z0-9_$#]+)"?$',
        expression
    )

    if not match:

        return None, None

    owner = normalize_identifier(
        match.group(1)
    )

    column = normalize_identifier(
        match.group(2)
    )

    table = aliases.get(
        owner,
        owner
    )

    return table, column


def detect_aggregate(
    expression: str
) -> Optional[str]:

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
            normalize_identifier(
                match.group(1)
            ),
            normalize_identifier(
                match.group(2)
            ),
        )
        for match in re.finditer(
            r'"?([A-Za-z0-9_$#]+)"?'
            r'\."?([A-Za-z0-9_$#]+)"?',
            expression,
        )
    ]


def lookup_column_metadata(
    metadata: Dict[str, Any],
    table: Optional[str],
    column: Optional[str]
) -> Dict[str, Any]:

    if not table or not column:

        return {}

    table_meta = metadata.get(
        "tables",
        {}
    ).get(
        normalize_identifier(table),
        {}
    )

    return table_meta.get(
        "columns",
        {}
    ).get(
        normalize_identifier(column),
        {}
    )


def parse_outputs(
    structure: Dict[str, Any],
    metadata: Dict[str, Any]
) -> List[Dict[str, Any]]:

    outputs = []

    aliases = structure.get(
        "aliases",
        {}
    )

    for position, expression_text in enumerate(
        structure.get(
            "select_list",
            []
        )
    ):

        expression, alias = (
            split_expression_alias(
                expression_text
            )
        )

        aggregate = detect_aggregate(
            expression
        )

        refs = expression_references(
            expression
        )

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

            table, column = (
                parse_source_column(
                    expression,
                    aliases
                )
            )

            source_table = table
            source_column = column

        datatype = None

        column_meta = lookup_column_metadata(
            metadata,
            source_table,
            source_column
        )

        if column_meta:

            datatype = column_meta.get(
                "datatype"
            )

        output_name = (
            alias
            or source_column
            or f"EXPR_{position + 1}"
        )

        is_numeric = (
            datatype in NUMERIC_TYPES
            or aggregate in {
                "COUNT",
                "SUM",
                "AVG",
            }
        )

        is_date = (
            datatype in DATE_TYPES
        )

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

def reflect_database(
    connection
) -> Dict[str, Any]:

    metadata = {
        "tables": {},
        "constraints": [],
        "indexes": [],
        "views": [],
        "objects": [],
    }

    print("\nReflecting Oracle metadata...")

    # --------------------------------------------------------
    # Tables
    # --------------------------------------------------------

    rows = fetch_rows(
        connection,
        """
        SELECT table_name
        FROM user_tables
        ORDER BY table_name
        """
    )

    for row in rows:

        table = normalize_identifier(
            row["TABLE_NAME"]
        )

        metadata["tables"][table] = {
            "columns": {},
            "pk_columns": set(),
            "unique_columns": set(),
            "comments": "",
        }

    print(
        f"Tables       : "
        f"{len(metadata['tables'])}"
    )

    # --------------------------------------------------------
    # Columns
    # --------------------------------------------------------

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

        table = normalize_identifier(
            row["TABLE_NAME"]
        )

        column = normalize_identifier(
            row["COLUMN_NAME"]
        )

        if table not in metadata["tables"]:

            metadata["tables"][table] = {
                "columns": {},
                "pk_columns": set(),
                "unique_columns": set(),
                "comments": "",
            }

        metadata["tables"][table][
            "columns"
        ][column] = {
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

    # --------------------------------------------------------
    # Constraints
    # --------------------------------------------------------

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
            normalize_identifier(
                row["CONSTRAINT_NAME"]
            )
        ].append(
            normalize_identifier(
                row["COLUMN_NAME"]
            )
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

            metadata["tables"][table][
                "pk_columns"
            ].update(columns)

        elif constraint_type == "U":

            metadata["tables"][table][
                "unique_columns"
            ].update(columns)

    print(
        "Constraints  :",
        len(constraints)
    )

    # --------------------------------------------------------
    # Indexes
    # --------------------------------------------------------

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

    print(
        "Indexes      :",
        len(metadata["indexes"])
    )

    # --------------------------------------------------------
    # Views
    # --------------------------------------------------------

    metadata["views"] = fetch_rows(
        connection,
        """
        SELECT
            view_name,
            text
        FROM user_views
        """
    )

    print(
        "Views        :",
        len(metadata["views"])
    )

    # --------------------------------------------------------
    # Objects
    # --------------------------------------------------------

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

    print(
        "Objects      :",
        len(metadata["objects"])
    )

    # --------------------------------------------------------
    # Table comments
    # --------------------------------------------------------

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

            metadata["tables"][table][
                "comments"
            ] = row["COMMENTS"] or ""

    # --------------------------------------------------------
    # Column comments
    # --------------------------------------------------------

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
            and column
            in metadata["tables"][table]["columns"]
        ):

            metadata["tables"][table][
                "columns"
            ][column]["comments"] = (
                row["COMMENTS"] or ""
            )

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
        {
            "prefix":
                STATIC_PACKAGE_PREFIX
        },
    )

    return [
        normalize_identifier(
            row["OBJECT_NAME"]
        )
        for row in rows[
            :MAX_STATIC_PACKAGES
        ]
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
        {
            "name":
                package_name
        },
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

    matches = list(
        pattern.finditer(source)
    )

    for index, match in enumerate(
        matches
    ):

        member_type = match.group(
            1
        ).upper()

        member_name = normalize_identifier(
            match.group(2)
        )

        start = match.start()

        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(source)
        )

        body = source[
            start:end
        ]

        members.append({
            "type": member_type,
            "name": member_name,
            "source": body,
        })

        if len(members) >= (
            MAX_STATIC_PROCEDURES
        ):

            break

    return members

def read_static_business_layer(
    connection,
    metadata
) -> Dict[str, Any]:
    """
    Read proven business SQL from configured Oracle packages.

    Package and procedure/function structure are retained.
    Nested SELECTs remain inside their parent SQL statement.
    The returned structure is consumed by build_capability_model().
    """

    queries = []
    package_names = []

    packages = discover_static_packages(
        connection
    )

    for package_name in packages:
        
        print(
        "\nDEBUG PACKAGE:",
        package_name
        )

        package_names.append(
            package_name
        )

        package_source = read_package_body(
            connection,
            package_name
        )
        
        print(
        "DEBUG PACKAGE SOURCE LENGTH:",
        len(package_source or "")
        )

        if not package_source:
            continue

        members = discover_package_members(
            package_source
        )
        
        print(
        "DEBUG MEMBERS:",
        [
            (
            m.get("type"),
            m.get("name"),
            len(m.get("source", ""))
            )
        for m in members
        ]
        )

        for member in members:

            member_source = member.get(
                "source",
                ""
            )

            
            if not member_source:
                continue
            
            print(
                "\nDEBUG MEMBER:",
                member.get("name")
            )
            
            print(
            "DEBUG MEMBER SOURCE LENGTH:",
            len(
            member.get("source", "")
            )
            )
            
               

            sql_statements = extract_select_statements(
                member_source
            )
            
            print(
                        "DEBUG SELECT COUNT:",
                        len(sql_statements)
                        )        
            
            for debug_sql in sql_statements:

                print("\nDEBUG SQL START")
                print(debug_sql[:3000])
                print("DEBUG SQL END")

    debug_structure = parse_select_structure(
        debug_sql
    )

    print(
        "DEBUG STRUCTURE:",
        bool(debug_structure)
    )

    if debug_structure:

        print(
            "DEBUG TABLES:",
            debug_structure.get(
                "tables",
                []
            )
        )

        print(
            "DEBUG SELECT LIST:",
            debug_structure.get(
                "select_list",
                []
            )
        )

        print(
            "DEBUG FROM:",
            debug_structure.get(
                "from",
                ""
            )[:500]
        )

        debug_outputs = parse_outputs(
            debug_structure,
            metadata
        )

        print(
            "DEBUG OUTPUT COUNT:",
            len(debug_outputs)
        )
                
        for sql in sql_statements:

                if not sql:
                    continue

                structure = parse_select_structure(
                    sql
                )

                if not structure:
                    continue

                outputs = parse_outputs(
                    structure,
                    metadata
                )

                queries.append({
                    "package":
                        package_name,

                    "member":
                        member.get(
                            "name",
                            ""
                        ),

                    "member_type":
                        member.get(
                            "type",
                            ""
                        ),

                    "sql":
                        sql,

                    "tables":
                        structure.get(
                            "tables",
                            []
                        ),

                    "aliases":
                        structure.get(
                            "aliases",
                            {}
                        ),

                    "joins":
                        structure.get(
                            "joins",
                            []
                        ),

                    "column_references":
                        structure.get(
                            "column_references",
                            []
                        ),

                    "select_list":
                        structure.get(
                            "select_list",
                            []
                        ),

                    "from":
                        structure.get(
                            "from",
                            ""
                        ),

                    "where":
                        structure.get(
                            "where",
                            ""
                        ),

                    "group_by":
                        structure.get(
                            "group_by",
                            []
                        ),

                    "having":
                        structure.get(
                            "having",
                            ""
                        ),

                    "order_by":
                        structure.get(
                            "order_by",
                            []
                        ),

                    "outputs":
                        outputs,
                })

    print(
        "\nStatic business packages:",
        len(package_names)
    )

    print(
        "Business SQL            :",
        len(queries)
    )

    return {
        "packages":
            package_names,

        "queries":
            queries,
    }


# ============================================================
# DATE EVIDENCE
# ============================================================

def discover_date_candidates(
    query: Dict[str, Any],
    metadata: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Date evidence comes from:

      1. SELECT outputs
      2. Columns referenced by the proven SQL
      3. Columns referenced by WHERE
      4. Known Oracle DATE/TIMESTAMP metadata

    No business-specific date column name is assumed.
    """

    candidates = []

    seen = set()

    # --------------------------------------------------------
    # SELECT outputs
    # --------------------------------------------------------

    for output in query.get(
        "outputs",
        []
    ):

        if output.get("is_date"):

            key = (
                output.get(
                    "source_table"
                ),
                output.get(
                    "source_column"
                )
            )

            if key not in seen:

                seen.add(key)

                candidates.append({
                    **output,
                    "evidence":
                        "SELECT_OUTPUT",
                })

    aliases = query.get(
        "aliases",
        {}
    )

    # --------------------------------------------------------
    # All qualified references.
    # --------------------------------------------------------

    for alias, column in query.get(
        "column_references",
        []
    ):

        table = aliases.get(
            alias,
            alias
        )

        column_meta = lookup_column_metadata(
            metadata,
            table,
            column
        )

        if (
            column_meta
            and column_meta.get(
                "datatype"
            ) in DATE_TYPES
        ):

            key = (
                normalize_identifier(table),
                normalize_identifier(column)
            )

            if key in seen:

                continue

            seen.add(key)

            candidates.append({
                "position": None,
                "alias": column,
                "expression":
                    f"{alias}.{column}",
                "source_table":
                    normalize_identifier(table),
                "source_column":
                    normalize_identifier(column),
                "datatype":
                    column_meta.get(
                        "datatype"
                    ),
                "is_numeric": False,
                "is_date": True,
                "aggregate": None,
                "references": [
                    (alias, column)
                ],
                "evidence":
                    "SQL_REFERENCE",
            })

    return candidates


# ============================================================
# CAPABILITY MODEL
# ============================================================

def build_capability_model(
    static_layer: Dict[str, Any],
    metadata: Dict[str, Any]
) -> List[Dict[str, Any]]:

    capabilities = []

    for query_id, query in enumerate(
        static_layer.get(
            "queries",
            []
        ),
        start=1
    ):

        outputs = query.get(
            "outputs",
            []
        )

        dimensions = []
        measures = []
        entity_keys = []

        # ----------------------------------------------------
        # Output classification
        # ----------------------------------------------------

        for output in outputs:

            source_table = normalize_identifier(
                output.get(
                    "source_table"
                ) or ""
            )

            source_column = normalize_identifier(
                output.get(
                    "source_column"
                ) or ""
            )

            table_meta = metadata.get(
                "tables",
                {}
            ).get(
                source_table,
                {}
            )

            pk_columns = table_meta.get(
                "pk_columns",
                set()
            )

            unique_columns = table_meta.get(
                "unique_columns",
                set()
            )

            is_entity_key = (
                source_column
                and (
                    source_column
                    in pk_columns
                    or source_column
                    in unique_columns
                )
            )

            # ------------------------------------------------
            # Entity key
            # ------------------------------------------------

            if is_entity_key:

                entity_keys.append(
                    output
                )

            # ------------------------------------------------
            # Numeric measures
            #
            # CRITICAL:
            # Entity keys are NOT measures.
            # ------------------------------------------------

            if (
                output.get("is_numeric")
                and not output.get("aggregate")
                and not is_entity_key
            ):

                measures.append(
                    output
                )

            # ------------------------------------------------
            # Natural dimensions
            # ------------------------------------------------

            if (
                not output.get("is_numeric")
                and not output.get("is_date")
            ):

                dimensions.append(
                    output
                )

        # ----------------------------------------------------
        # GROUP BY strengthens dimension evidence.
        # ----------------------------------------------------

        group_by = query.get(
            "group_by",
            []
        )

        for expression in group_by:

            refs = expression_references(
                expression
            )

            for _, column in refs:

                column = normalize_identifier(
                    column
                )

                for output in outputs:

                    if (
                        normalize_identifier(
                            output.get(
                                "source_column"
                            ) or ""
                        )
                        == column
                    ):

                        if output not in dimensions:

                            dimensions.append(
                                output
                            )

        # ----------------------------------------------------
        # Date evidence.
        # ----------------------------------------------------

        date_candidates = (
            discover_date_candidates(
                query,
                metadata
            )
        )

        # ----------------------------------------------------
        # Grain confidence.
        # ----------------------------------------------------

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
            "capability_id":
                query_id,

            "package":
                query["package"],

            "member":
                query["member"],

            "sql":
                query["sql"],

            "tables":
                query["tables"],

            "aliases":
                query["aliases"],

            "joins":
                query["joins"],

            "outputs":
                outputs,

            "dimensions":
                dimensions,

            "measures":
                measures,

            "entity_keys":
                entity_keys,

            "date_candidates":
                date_candidates,

            "filters":
                query.get(
                    "where",
                    ""
                ),

            "group_by":
                query.get(
                    "group_by",
                    []
                ),

            "having":
                query.get(
                    "having",
                    ""
                ),

            "order_by":
                query.get(
                    "order_by",
                    []
                ),

            "column_references":
                query.get(
                    "column_references",
                    []
                ),

            "grain_score":
                min(
                    grain_score,
                    1.0
                ),
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


def extract_time_range(
    text: str
) -> Dict[str, Any]:

    text_lower = text.lower()

    # --------------------------------------------------------
    # Month + year
    # --------------------------------------------------------

    for (
        month_name,
        month_number
    ) in MONTHS.items():

        pattern = (
            rf"\b{month_name}\s+(\d{{4}})\b"
        )

        match = re.search(
            pattern,
            text_lower
        )

        if match:

            year = int(
                match.group(1)
            )

            start = (
                f"{year:04d}-"
                f"{month_number:02d}-01"
            )

            if month_number == 12:

                next_year = year + 1
                next_month = 1

            else:

                next_year = year
                next_month = (
                    month_number + 1
                )

            end = (
                f"{next_year:04d}-"
                f"{next_month:02d}-01"
            )

            return {
                "type":
                    "month",

                "label":
                    f"{month_name} {year}",

                "start":
                    start,

                "end":
                    end,
            }

    # --------------------------------------------------------
    # ISO date
    # --------------------------------------------------------

    match = re.search(
        r"\b(\d{4}-\d{2}-\d{2})\b",
        text_lower
    )

    if match:

        date_value = match.group(
            1
        )

        return {
            "type":
                "date",

            "label":
                date_value,

            "start":
                date_value,

            "end":
                date_value,
        }

    return {}


def analyse_prompt(
    prompt: str
) -> Dict[str, Any]:

    tokens = tokenize(prompt)

    meaningful = []

    for token in tokens:

        if token in STOPWORDS:

            continue

        meaningful.append(
            token
        )

    operation = None

    for token in tokens:

        if token in OPERATION_WORDS:

            operation = (
                OPERATION_WORDS[token]
            )

            break

    dimension_requested = any(
        token in DIMENSION_WORDS
        for token in tokens
    )

    time_range = extract_time_range(
        prompt
    )

    dimension_terms = []

    # --------------------------------------------------------
    # "region wise"
    # --------------------------------------------------------

    wise_match = re.search(
        r"\b([a-zA-Z][a-zA-Z0-9_]*)\s+wise\b",
        prompt,
        re.I
    )

    if wise_match:

        dimension_terms.append(
            wise_match.group(1).lower()
        )

    # --------------------------------------------------------
    # "by region"
    # --------------------------------------------------------

    by_match = re.search(
        r"\bby\s+([a-zA-Z][a-zA-Z0-9_]*)",
        prompt,
        re.I
    )

    if by_match:

        dimension_terms.append(
            by_match.group(1).lower()
        )

    # --------------------------------------------------------
    # "per region"
    # --------------------------------------------------------

    per_match = re.search(
        r"\bper\s+([a-zA-Z][a-zA-Z0-9_]*)",
        prompt,
        re.I
    )

    if per_match:

        dimension_terms.append(
            per_match.group(1).lower()
        )

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

    for dimension_term in dimension_terms:

        concepts = [
            x
            for x in concepts
            if x != dimension_term
        ]

    return {
        "prompt":
            prompt,

        "tokens":
            tokens,

        "concept_terms":
            concepts,

        "subject_terms":
            concepts,

        "measure_terms":
            concepts,

        "dimension_terms":
            dimension_terms,

        "dimension_requested":
            dimension_requested,

        "operation_requested":
            operation,

        "time_range":
            time_range,
    }


# ============================================================
# CAPABILITY TEXT
# ============================================================

def capability_text(
    capability: Dict[str, Any]
) -> str:

    parts = [
        capability.get(
            "package",
            ""
        ),
        capability.get(
            "member",
            ""
        ),
    ]

    for output in capability.get(
        "outputs",
        []
    ):

        parts.extend([
            output.get(
                "alias",
                ""
            ),
            output.get(
                "source_column",
                ""
            ) or "",
            output.get(
                "source_table",
                ""
            ) or "",
            output.get(
                "expression",
                ""
            ),
        ])

    for table in capability.get(
        "tables",
        []
    ):

        parts.append(table)

    # SQL itself is semantic evidence.
    parts.append(
        capability.get(
            "sql",
            ""
        )
    )

    return " ".join(
        parts
    ).lower()


def score_term_against_output(
    term: str,
    output: Dict[str, Any]
) -> float:

    candidates = [
        output.get(
            "alias",
            ""
        ),
        output.get(
            "source_column",
            ""
        ) or "",
        output.get(
            "source_table",
            ""
        ) or "",
        output.get(
            "expression",
            ""
        ),
    ]

    best = 0.0

    term_forms = word_forms(
        term
    )

    for candidate in candidates:

        candidate_parts = set(
            identifier_parts(
                candidate
            )
        )

        candidate_text = (
            candidate or ""
        ).lower()

        for form in term_forms:

            if form in candidate_parts:

                best = max(
                    best,
                    1.0
                )

            elif form in candidate_text:

                best = max(
                    best,
                    0.85
                )

            else:

                for part in candidate_parts:

                    best = max(
                        best,
                        similarity(
                            form,
                            part
                        )
                    )

    return min(
        best,
        1.0
    )


def score_term_against_capability(
    term: str,
    capability: Dict[str, Any]
) -> float:
    """
    Capability-level semantic evidence.

    This deliberately does not create a business vocabulary
    dictionary.

    A term can match:

      - package
      - procedure/function
      - table
      - output
      - column
      - SQL text

    The engine therefore learns meaning from the existing
    application implementation rather than from a hardcoded
    dictionary.
    """

    best = 0.0

    # --------------------------------------------------------
    # Outputs
    # --------------------------------------------------------

    for output in capability.get(
        "outputs",
        []
    ):

        best = max(
            best,
            score_term_against_output(
                term,
                output
            )
        )

    # --------------------------------------------------------
    # Package / member / table names
    # --------------------------------------------------------

    names = [
        capability.get(
            "package",
            ""
        ),
        capability.get(
            "member",
            ""
        ),
    ]

    names.extend(
        capability.get(
            "tables",
            []
        )
    )

    names.extend([
        output.get(
            "alias",
            ""
        )
        for output in capability.get(
            "outputs",
            []
        )
    ])

    for name in names:

        for part in identifier_parts(
            name
        ):

            for form in word_forms(
                term
            ):

                if form == part:

                    best = max(
                        best,
                        1.0
                    )

                else:

                    best = max(
                        best,
                        similarity(
                            form,
                            part
                        ) * 0.75
                    )

    # --------------------------------------------------------
    # SQL semantic evidence
    # --------------------------------------------------------

    sql_text = normalize_space(
        capability.get(
            "sql",
            ""
        ).lower()
    )

    if sql_text:

        for form in word_forms(
            term
        ):

            if re.search(
                rf"\b{re.escape(form)}\b",
                sql_text,
                re.I
            ):

                best = max(
                    best,
                    0.65
                )

    return min(
        best,
        1.0
    )


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

        local = score_term_against_capability(
            term,
            capability
        )

        subject_score = max(
            subject_score,
            local
        )

    # --------------------------------------------------------
    # Measure
    #
    # ONLY numeric non-key measures are considered here.
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
    # Date
    # --------------------------------------------------------

    if request.get(
        "time_range"
    ):

        if capability.get(
            "date_candidates"
        ):

            time_score = 1.0

    # --------------------------------------------------------
    # Entity
    # --------------------------------------------------------

    if capability.get(
        "entity_keys"
    ):

        entity_score = 1.0

    elif capability.get(
        "outputs"
    ):

        entity_score = 0.40

    # --------------------------------------------------------
    # Completeness
    # --------------------------------------------------------

    completeness = 0.0

    if subject_score > 0:

        completeness += 0.30

    if request.get(
        "dimension_requested"
    ):

        if dimension_score > 0:

            completeness += 0.25

    else:

        completeness += 0.20

    if request.get(
        "time_range"
    ):

        if time_score > 0:

            completeness += 0.20

    else:

        completeness += 0.15

    if entity_score > 0:

        completeness += 0.10

    if capability.get(
        "joins"
    ):

        completeness += 0.05

    # --------------------------------------------------------
    # Overall score
    # --------------------------------------------------------

    score = (
        subject_score * 0.35
        + measure_score * 0.20
        + dimension_score * 0.15
        + time_score * 0.10
        + entity_score * 0.10
        + completeness * 0.10
    )

    return {
        "capability":
            capability,

        "score":
            round(score, 4),

        "subject_score":
            round(subject_score, 4),

        "measure_score":
            round(measure_score, 4),

        "dimension_score":
            round(dimension_score, 4),

        "time_score":
            round(time_score, 4),

        "entity_score":
            round(entity_score, 4),

        "completeness":
            round(completeness, 4),
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

    return scored[
        :MAX_CANDIDATES
    ]


# ============================================================
# OUTPUT RESOLUTION
# ============================================================

def resolve_output(
    capability: Dict[str, Any],
    terms: List[str],
    mode: str
) -> Optional[Dict[str, Any]]:

    if mode == "dimension":

        candidates = capability.get(
            "dimensions",
            []
        )

    elif mode == "measure":

        candidates = capability.get(
            "measures",
            []
        )

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

    else:

        candidates = capability.get(
            "outputs",
            []
        )

    if not candidates:

        return None

    # --------------------------------------------------------
    # Explicit terms
    # --------------------------------------------------------

    if terms:

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

        if best is not None:

            return best

    # --------------------------------------------------------
    # No explicit term:
    #
    # use strongest structural evidence.
    # --------------------------------------------------------

    if mode == "entity":

        return candidates[0]

    if mode == "date":

        return candidates[0]

    return candidates[0]


# ============================================================
# QUERY PLAN
# ============================================================

def resolve_query_plan(
    match: Dict[str, Any],
    request: Dict[str, Any]
) -> Dict[str, Any]:

    capability = match[
        "capability"
    ]

    operation_requested = request.get(
        "operation_requested"
    )

    # --------------------------------------------------------
    # Entity
    # --------------------------------------------------------

    entity_key = resolve_output(
        capability,
        request.get(
            "subject_terms",
            []
        ),
        "entity"
    )

    # --------------------------------------------------------
    # Numeric measure
    #
    # Do NOT resolve a PK as a numeric measure.
    # --------------------------------------------------------

    measure = resolve_output(
        capability,
        request.get(
            "measure_terms",
            []
        ),
        "measure"
    )

    # --------------------------------------------------------
    # Dimension
    # --------------------------------------------------------

    dimension = resolve_output(
        capability,
        request.get(
            "dimension_terms",
            []
        ),
        "dimension"
    )

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    date_output = resolve_output(
        capability,
        [],
        "date"
    )

    # --------------------------------------------------------
    # Operation resolution
    # --------------------------------------------------------

    if operation_requested == "TOTAL":

        if measure:

            # TOTAL applied to a proven numeric measure.
            operation = "SUM"

        elif entity_key:

            # TOTAL applied to an entity means count.
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

        elif entity_key:

            operation = "COUNT"

        else:

            operation = "SELECT"

    return {
        "intent":
            (
                "aggregate"
                if operation != "SELECT"
                else "select"
            ),

        "operation":
            operation,

        "subject":
            entity_key,

        "measure":
            measure,

        "dimension":
            dimension,

        "entity_key":
            entity_key,

        "date":
            date_output,

        "time_range":
            request.get(
                "time_range"
            ),

        "capability":
            capability,

        "match_score":
            match["score"],
    }


# ============================================================
# SQL IDENTIFIER SAFETY
# ============================================================

def quote_identifier(
    identifier: str
) -> str:

    identifier = normalize_identifier(
        identifier
    )

    if not re.match(
        r"^[A-Z0-9_$#]+$",
        identifier
    ):

        raise ValueError(
            f"Unsafe SQL identifier: "
            f"{identifier}"
        )

    return f'"{identifier}"'


# ============================================================
# DATE FILTER COMPOSITION
# ============================================================

def build_date_predicate(
    date_output: Dict[str, Any],
    time_range: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:

    source_column = normalize_identifier(
        date_output.get(
            "source_column"
        ) or ""
    )

    if not source_column:

        raise ValueError(
            "A proven date column could not "
            "be resolved."
        )

    # The date expression is selected from proven SQL
    # evidence, not a hardcoded business mapping.
    #
    # Bind variables are used so the generated SQL remains
    # safe and reusable.
    predicate = (
        f'Q."{source_column}" >= '
        f"TO_DATE(:p_date_start, "
        f"'YYYY-MM-DD') "
        f'AND Q."{source_column}" < '
        f"TO_DATE(:p_date_end, "
        f"'YYYY-MM-DD')"
    )

    params = {
        "p_date_start":
            time_range["start"],

        "p_date_end":
            time_range["end"],
    }

    return predicate, params


# ============================================================
# STATIC QUERY COMPOSITION
# ============================================================

def compose_query(
    plan: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:

    capability = plan["capability"]

    # --------------------------------------------------------
    # IMPORTANT:
    # Preserve the proven business SQL exactly as supplied.
    #
    # Do NOT normalize whitespace here because the SQL may
    # contain -- comments whose meaning depends on line breaks.
    # --------------------------------------------------------

    base_sql = (
        capability["sql"]
        .strip()
        .rstrip(";")
        .strip()
    )

    operation = plan["operation"]

    entity_key = plan.get(
        "entity_key"
    )

    measure = plan.get(
        "measure"
    )

    dimension = plan.get(
        "dimension"
    )

    date_output = plan.get(
        "date"
    )

    time_range = plan.get(
        "time_range"
    )

    params = {}

    # ========================================================
    # DATE VALIDATION
    # ========================================================

    if time_range and not date_output:

        raise ValueError(
            "The request contains a time range, "
            "but the selected proven business capability "
            "contains no resolvable DATE/TIMESTAMP evidence."
        )

    # ========================================================
    # DATE FILTER
    #
    # IMPORTANT:
    # We are NOT modifying the proven business SQL.
    #
    # The date filter is applied outside the proven query.
    # ========================================================

    date_where = ""

    if time_range and date_output:

        date_alias = normalize_identifier(
            date_output.get("alias")
            or date_output.get("source_column")
            or ""
        )

        if not date_alias:

            raise ValueError(
                "The selected date evidence has no "
                "resolvable output alias."
            )

        # ----------------------------------------------------
        # Make sure the date column is actually exposed by
        # the proven business SQL.
        # ----------------------------------------------------

        output_aliases = {
            normalize_identifier(
                output.get("alias", "")
            )
            for output in capability.get(
                "outputs",
                []
            )
            if output.get("alias")
        }

        if date_alias not in output_aliases:

            raise ValueError(
                "The proven business SQL uses date evidence "
                "but the selected date column is not exposed "
                "by the proven query. "
                "A safe generic outer date filter cannot "
                "be applied without changing the proven SQL."
            )

        # ----------------------------------------------------
        # Build the actual date range.
        #
        # We use the existing time_range values instead of
        # inventing another date representation.
        # ----------------------------------------------------

        start_date = time_range.get(
            "start"
        )

        end_date = time_range.get(
            "end"
        )

        if not start_date or not end_date:

            raise ValueError(
                "The resolved time range does not contain "
                "both start and end dates."
            )

        params["p_date_start"] = start_date
        params["p_date_end"] = end_date

        date_where = f"""
WHERE
    Q."{date_alias}" >= TO_DATE(
        :p_date_start,
        'YYYY-MM-DD'
    )
    AND Q."{date_alias}" < TO_DATE(
        :p_date_end,
        'YYYY-MM-DD'
    )
"""

    # ========================================================
    # IDENTIFIERS
    # ========================================================

    entity_alias = None

    if entity_key:

        entity_alias = quote_identifier(
            entity_key["alias"]
        )

    measure_alias = None

    if measure:

        measure_alias = quote_identifier(
            measure["alias"]
        )

    dimension_alias = None

    if dimension:

        dimension_alias = quote_identifier(
            dimension["alias"]
        )

    # ========================================================
    # AGGREGATION
    # ========================================================

    if operation == "COUNT":

        if entity_alias:

            select_expression = (
                f"COUNT(DISTINCT "
                f"{entity_alias})"
            )

        else:

            select_expression = "COUNT(*)"

        sql = f"""
SELECT
    {select_expression} AS RESULT_COUNT
FROM (
    {base_sql}
) Q
{date_where}
"""

    elif operation == "SUM":

        if not measure_alias:

            raise ValueError(
                "SUM requested but no proven "
                "numeric measure was resolved."
            )

        sql = f"""
SELECT
    SUM({measure_alias}) AS RESULT_TOTAL
FROM (
    {base_sql}
) Q
{date_where}
"""

    elif operation == "AVG":

        if not measure_alias:

            raise ValueError(
                "AVG requested but no proven "
                "numeric measure was resolved."
            )

        sql = f"""
SELECT
    AVG({measure_alias}) AS RESULT_AVERAGE
FROM (
    {base_sql}
) Q
{date_where}
"""

    elif operation == "MIN":

        if not measure_alias:

            raise ValueError(
                "MIN requested but no proven "
                "numeric measure was resolved."
            )

        sql = f"""
SELECT
    MIN({measure_alias}) AS RESULT_MIN
FROM (
    {base_sql}
) Q
{date_where}
"""

    elif operation == "MAX":

        if not measure_alias:

            raise ValueError(
                "MAX requested but no proven "
                "numeric measure was resolved."
            )

        sql = f"""
SELECT
    MAX({measure_alias}) AS RESULT_MAX
FROM (
    {base_sql}
) Q
{date_where}
"""

    else:

        sql = f"""
SELECT *
FROM (
    {base_sql}
) Q
{date_where}
"""

    # ========================================================
    # DIMENSION
    # ========================================================

    if dimension_alias:

        if operation == "COUNT":

            if entity_alias:

                aggregation = (
                    f"COUNT(DISTINCT "
                    f"{entity_alias})"
                )

            else:

                aggregation = "COUNT(*)"

            select_list = (
                f"{dimension_alias} "
                f"AS DIMENSION_VALUE,\n"
                f"{aggregation} "
                f"AS RESULT_COUNT"
            )

        elif operation == "SUM":

            if not measure_alias:

                raise ValueError(
                    "SUM by dimension requested but "
                    "no proven numeric measure was resolved."
                )

            select_list = (
                f"{dimension_alias} "
                f"AS DIMENSION_VALUE,\n"
                f"SUM({measure_alias}) "
                f"AS RESULT_TOTAL"
            )

        elif operation == "AVG":

            if not measure_alias:

                raise ValueError(
                    "AVG by dimension requested but "
                    "no proven numeric measure was resolved."
                )

            select_list = (
                f"{dimension_alias} "
                f"AS DIMENSION_VALUE,\n"
                f"AVG({measure_alias}) "
                f"AS RESULT_AVERAGE"
            )

        elif operation == "MIN":

            if not measure_alias:

                raise ValueError(
                    "MIN by dimension requested but "
                    "no proven numeric measure was resolved."
                )

            select_list = (
                f"{dimension_alias} "
                f"AS DIMENSION_VALUE,\n"
                f"MIN({measure_alias}) "
                f"AS RESULT_MIN"
            )

        elif operation == "MAX":

            if not measure_alias:

                raise ValueError(
                    "MAX by dimension requested but "
                    "no proven numeric measure was resolved."
                )

            select_list = (
                f"{dimension_alias} "
                f"AS DIMENSION_VALUE,\n"
                f"MAX({measure_alias}) "
                f"AS RESULT_MAX"
            )

        else:

            select_list = "*"

        sql = f"""
SELECT
    {select_list}
FROM (
    {base_sql}
) Q
{date_where}
"""

        if operation != "SELECT":

            sql += f"""
GROUP BY
    {dimension_alias}
ORDER BY
    {dimension_alias}
"""

    # ========================================================
    # FINAL RETURN
    # ========================================================

    return sql.strip(), params

# ============================================================
# OPTIONAL RUNTIME SQL HISTORY
# ============================================================

def read_runtime_sql_history(
    connection
) -> List[Dict[str, Any]]:

    """
    Optional runtime evidence.

    This is deliberately failure-safe because access to
    V$SQL depends on Oracle privileges.

    Runtime history is supplementary evidence.
    Proven package SQL remains primary evidence.
    """

    if not ENABLE_RUNTIME_SQL_HISTORY:

        return []

    try:

        rows = fetch_rows(
            connection,
            """
            SELECT *
            FROM (
                SELECT
                    sql_id,
                    executions,
                    last_active_time,
                    sql_text
                FROM v$sql
                WHERE sql_text IS NOT NULL
                  AND (
                      UPPER(sql_text)
                      LIKE '%SELECT%'
                  )
                ORDER BY
                    last_active_time DESC
            )
            WHERE ROWNUM <= :max_rows
            """,
            {
                "max_rows":
                    MAX_HISTORY_ROWS
            }
        )

        return rows

    except Exception as exc:

        print(
            "\nRuntime SQL history unavailable:"
        )

        print(
            str(exc)
        )

        return []


# ============================================================
# ORACLE VALIDATION
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
            str(
                row.get(
                    "PLAN_TABLE_OUTPUT",
                    ""
                )
            )
            for row in rows
        )

        return {
            "valid":
                True,

            "plan":
                plan_text,
        }

    except Exception as exc:

        return {
            "valid":
                False,

            "error":
                str(exc),

            "plan":
                "",
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
        "columns":
            columns,

        "rows":
            rows,

        "row_count":
            len(rows),
    }


# ============================================================
# DISPLAY
# ============================================================

def print_request(
    request: Dict[str, Any]
):

    print(
        "\n" + "=" * 70
    )

    print(
        "REQUEST PLAN"
    )

    print(
        "=" * 70
    )

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

    print(
        "\n" + "=" * 70
    )

    print(
        "CAPABILITY MATCHES"
    )

    print(
        "=" * 70
    )

    for index, match in enumerate(
        matches[:10],
        start=1
    ):

        capability = match[
            "capability"
        ]

        print(
            f"\n[{index}] "
            f"{capability['package']}."
            f"{capability['member']}"
        )

        print(
            f"    Score      : "
            f"{match['score']}"
        )

        print(
            f"    Subject    : "
            f"{match['subject_score']}"
        )

        print(
            f"    Measure    : "
            f"{match['measure_score']}"
        )

        print(
            f"    Dimension  : "
            f"{match['dimension_score']}"
        )

        print(
            f"    Time       : "
            f"{match['time_score']}"
        )

        print(
            f"    Entity     : "
            f"{match['entity_score']}"
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

        print(
            "    Keys       :",
            [
                x["alias"]
                for x in capability.get(
                    "entity_keys",
                    []
                )
            ]
        )

        print(
            "    Measures   :",
            [
                x["alias"]
                for x in capability.get(
                    "measures",
                    []
                )
            ]
        )

        print(
            "    Dimensions :",
            [
                x["alias"]
                for x in capability.get(
                    "dimensions",
                    []
                )
            ]
        )

        print(
            "    Dates      :",
            [
                x.get(
                    "alias"
                )
                for x in capability.get(
                    "date_candidates",
                    []
                )
            ]
        )


def print_plan(
    plan: Dict[str, Any]
):

    print(
        "\n" + "=" * 70
    )

    print(
        "RESOLVED QUERY PLAN"
    )

    print(
        "=" * 70
    )

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

    print(
        "\n" + "=" * 70
    )

    print(
        "RESULT"
    )

    print(
        "=" * 70
    )

    columns = result[
        "columns"
    ]

    print(
        "Columns:",
        columns
    )

    for row in result[
        "rows"
    ][:100]:

        print(row)

    if result[
        "row_count"
    ] > 100:

        print(
            f"... "
            f"{result['row_count'] - 100} "
            f"more rows"
        )


# ============================================================
# COMPLETE PROCESS
# ============================================================

def process_prompt(
    connection,
    prompt: str,
    metadata: Dict[str, Any],
    capabilities: List[Dict[str, Any]]
):

    request = analyse_prompt(
        prompt
    )

    print_request(
        request
    )

    if not capabilities:

        raise RuntimeError(
            "No static business SQL "
            "capabilities were discovered."
        )

    matches = match_capabilities(
        capabilities,
        request
    )

    print_capability_matches(
        matches
    )

    if not matches:

        raise RuntimeError(
            "No business capability "
            "matched the request."
        )

    best = matches[0]

    if best["score"] < (
        MIN_CAPABILITY_SCORE
    ):

        raise RuntimeError(
            "Business meaning could not be "
            "resolved with sufficient evidence."
        )

    plan = resolve_query_plan(
        best,
        request
    )

    print_plan(
        plan
    )

    sql, params = compose_query(
        plan
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "GENERATED SQL"
    )

    print(
        "=" * 70
    )

    print(sql)

    if params:

        print(
            "\nBind parameters:",
            params
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "ORACLE VALIDATION"
    )

    print(
        "=" * 70
    )

    validation = validate_sql(
        connection,
        sql,
        params
    )

    if not validation[
        "valid"
    ]:

        print(
            "VALIDATION FAILED"
        )

        print(
            validation[
                "error"
            ]
        )

        return {
            "request":
                request,

            "plan":
                plan,

            "sql":
                sql,

            "validation":
                validation,

            "result":
                None,
        }

    print(
        "SQL VALID"
    )

    print(
        "\n" + validation[
            "plan"
        ]
    )

    # --------------------------------------------------------
    # Execute only after successful validation.
    # --------------------------------------------------------

    try:

        result = run_query(
            connection,
            sql,
            params
        )

        print_results(
            result
        )

    except Exception as exc:

        print(
            "\nEXECUTION FAILED"
        )

        print(
            str(exc)
        )

        result = {
            "error":
                str(exc)
        }

    return {
        "request":
            request,

        "plan":
            plan,

        "sql":
            sql,

        "validation":
            validation,

        "result":
            result,
    }


# ============================================================
# CAPABILITY REPORT
# ============================================================

def print_capability_report(
    capabilities: List[Dict[str, Any]]
):

    print(
        "\n" + "=" * 70
    )

    print(
        "STATIC BUSINESS CAPABILITY MODEL"
    )

    print(
        "=" * 70
    )

    for capability in capabilities:

        print(
            f"\nCapability #"
            f"{capability['capability_id']}"
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

        print(
            "Dates:",
            [
                x.get(
                    "alias"
                )
                for x in capability.get(
                    "date_candidates",
                    []
                )
            ]
        )


# ============================================================
# INTERACTIVE ENGINE
# ============================================================

def run_lab():

    print(
        "=" * 70
    )

    print(
        "DYNAMIC ORACLE BUSINESS "
        "INTELLIGENCE ENGINE"
    )

    print(
        "=" * 70
    )

    print(
        f"Database: {DATABASE_NAME}"
    )

    connection = None

    try:

        # ----------------------------------------------------
        # Connect once.
        # ----------------------------------------------------

        print(
            "\nConnecting to Oracle..."
        )

        connection = get_db_connection(
            DATABASE_NAME
        )

        print(
            "Connected."
        )

        # ----------------------------------------------------
        # Reflect once.
        #
        # This is startup intelligence, not per-prompt
        # reflection.
        # ----------------------------------------------------

        metadata = reflect_database(
            connection
        )

        # ----------------------------------------------------
        # Read proven business SQL once.
        # ----------------------------------------------------

        static_layer = read_static_business_layer(
        connection,
        metadata    
        )

        # ----------------------------------------------------
        # Build capability model once.
        # ----------------------------------------------------

        capabilities = (
            build_capability_model(
                static_layer,
                metadata
            )
        )

        print(
            "\nCapabilities  :",
            len(capabilities)
        )

        print_capability_report(
            capabilities
        )

        # ----------------------------------------------------
        # Optional runtime evidence.
        # ----------------------------------------------------

        history = (
            read_runtime_sql_history(
                connection
            )
        )

        print(
            "\nRuntime SQL history:",
            len(history)
        )

        # ----------------------------------------------------
        # Interactive prompt.
        # ----------------------------------------------------

        print(
            "\n" + "=" * 70
        )

        print(
            "INTERACTIVE MODE"
        )

        print(
            "=" * 70
        )

        print(
            "Enter a natural-language business question."
        )

        print(
            "Type EXIT or QUIT to stop."
        )

        while True:

            try:

                prompt = input(
                    "\nBusiness question > "
                ).strip()

            except (
                EOFError,
                KeyboardInterrupt
            ):

                print(
                    "\nStopping engine..."
                )

                break

            if not prompt:

                continue

            if prompt.lower() in {
                "exit",
                "quit",
            }:

                print(
                    "\nStopping engine..."
                )

                break

            try:

                process_prompt(
                    connection,
                    prompt,
                    metadata,
                    capabilities
                )

            except Exception as exc:

                print(
                    "\nENGINE ERROR:"
                )

                print(
                    str(exc)
                )

    finally:

        if connection:

            connection.close()

        print(
            "\nConnection closed."
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_lab()

