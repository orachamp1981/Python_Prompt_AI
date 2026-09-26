"""
Step 1 — Extraction Layer
Pulls a single package's source, splits it into per-procedure blocks,
pulls its schema-level dependencies and argument list, flags any
database-link references, and saves everything to a local JSON store.

No interpretation happens here — this is raw material collection only.
The output feeds Step 2 (cleaning) and Step 3 (LLM extraction) later.
"""

import re
import json
from pathlib import Path
from datetime import datetime

from databases import get_db_connection

# Where extracted packages get saved, one folder per package
STORE_DIR = Path("extracted_packages")
STORE_DIR.mkdir(exist_ok=True)


def parse_column_tags(source_text: str) -> list:
    """
    Scans a procedure's source text for column-level tags and extracts
    them into structured records. Handles two tag styles:

    1. Plain trailing @label: tag — a pipe-separated list of business
       synonyms for a stored column, e.g.:
           DOCUMENT_CODE,  -- @label: Policy Number | Policy | Document No

    2. Computed-field block — for CASE statements / derived values,
       spread across up to three comment lines directly after the
       expression:
           END INVEST_STATUS
           -- @concept: INVEST_STATUS
           -- @derived: true
           -- @logic: Current if ISSUANCE_DATE falls within ...

    Returns a list of dicts, one per tagged item. Untagged columns are
    simply absent from the result — nothing is guessed on their behalf.
    """
    tags = []

    # --- Style 1: trailing @label: lists on a single line ---
    # Captures the column name preceding the comma/whitespace before
    # the comment, and the pipe-separated labels after @label:
    label_pattern = re.compile(
        r"([A-Za-z_][A-Za-z0-9_]*)\s*,?\s*--\s*@label:\s*(.+)$",
        re.MULTILINE,
    )
    for match in label_pattern.finditer(source_text):
        column_name = match.group(1).upper()
        raw_labels = match.group(2)
        # Split on |, strip whitespace, drop empty entries (handles the
        # trailing "| " case cleanly without requiring manual cleanup)
        labels = [l.strip() for l in raw_labels.split("|")]
        labels = [l for l in labels if l]
        if labels:
            tags.append({
                "column": column_name,
                "tag_type": "label",
                "labels": labels,
            })

    # --- Style 2: @concept / @derived / @logic block for computed fields ---
    concept_pattern = re.compile(
        r"@concept:\s*(\S+)\s*\n\s*--\s*@derived:\s*(\w+)\s*\n\s*--\s*@logic:\s*(.+)",
        re.IGNORECASE,
    )
    for match in concept_pattern.finditer(source_text):
        tags.append({
            "column": match.group(1).upper(),
            "tag_type": "derived_concept",
            "concept": match.group(1).upper(),
            "derived": match.group(2).strip().lower() == "true",
            "logic": match.group(3).strip(),
        })

    return tags


def parse_procedure_label(proc_source: str) -> str:
    """
    Looks for a procedure-level @label: tag — one that sits alone on its
    own comment line (not trailing after a column name), e.g.:
        -- @label: MBS Business Logic Initiator
        OPEN p_result FOR ...
    This is distinct from column-level @label: tags, which always have
    a column name before the "--" on the same line. Returns the first
    standalone match found, or None if not tagged.
    """
    standalone_pattern = re.compile(
        r"^\s*--\s*@label:\s*(.+)$",
        re.MULTILINE,
    )
    match = standalone_pattern.search(proc_source)
    if match:
        return match.group(1).strip()
    return None


def get_package_source(conn, package_name: str, database_name: str) -> str:
    """
    Pulls ALL_SOURCE for the package (both SPEC and BODY), ordered by
    line number, and reassembles it into one readable text block.
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT type, line, text
        FROM ALL_SOURCE
        WHERE owner = :owner
          AND name = :name
          AND type IN ('PACKAGE', 'PACKAGE BODY')
        ORDER BY type, line
        """,
        owner=database_name.upper(),
        name=package_name.upper(),
    )
    rows = cursor.fetchall()
    cursor.close()

    if not rows:
        raise ValueError(
            f"No source found for package {package_name} under owner {database_name}. "
            f"Check the package exists and owner/schema is correct."
        )

    # Join lines back together, keeping spec and body separate for clarity
    spec_lines = [text for (t, _, text) in rows if t == "PACKAGE"]
    body_lines = [text for (t, _, text) in rows if t == "PACKAGE BODY"]

    full_text = "-- PACKAGE SPEC --\n" + "".join(spec_lines)
    full_text += "\n\n-- PACKAGE BODY --\n" + "".join(body_lines)
    return full_text


def split_into_procedures(source_text: str) -> dict:
    """
    Splits reassembled package body source into individual procedure/
    function blocks. Matches from 'PROCEDURE <name>' or 'FUNCTION <name>'
    down to the matching 'END <name>;'.

    This is a pragmatic regex-based split, not a full PL/SQL parser —
    it works well for standard formatting but can miss edge cases
    (e.g. nested procedures, unusual END styles). Flag anything that
    looks off for manual review rather than trusting it blindly.
    """
    pattern = re.compile(
        r"(PROCEDURE|FUNCTION)\s+(\w+)\s*\(.*?END\s+\2\s*;",
        re.IGNORECASE | re.DOTALL,
    )

    procedures = {}
    for match in pattern.finditer(source_text):
        proc_type = match.group(1).upper()
        proc_name = match.group(2).upper()
        block_text = match.group(0)
        procedures[proc_name] = {
            "type": proc_type,
            "source": block_text,
        }

    return procedures


def get_dependencies(conn, package_name: str, database_name: str) -> list:
    """
    Pulls ALL_DEPENDENCIES for the package — every table, view, synonym,
    or other object it references. This is package-level, not per-procedure;
    Oracle's dictionary doesn't track dependencies at that granularity.
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT referenced_owner, referenced_name, referenced_type
        FROM ALL_DEPENDENCIES
        WHERE owner = :owner
          AND name = :name
          AND type = 'PACKAGE BODY'
        ORDER BY referenced_owner, referenced_name
        """,
        owner=database_name.upper(),
        name=package_name.upper(),
    )
    rows = cursor.fetchall()
    cursor.close()

    return [
        {"owner": r[0], "name": r[1], "type": r[2]}
        for r in rows
    ]


def get_table_columns(conn, owner: str, table_name: str) -> list:
    """
    Pulls column-level detail for one table/view — name, data type,
    nullability, and length/precision. Used to attach real structure
    to each dependency found for the package, instead of just a name.
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT column_name, data_type, data_length, data_precision,
               data_scale, nullable, column_id
        FROM ALL_TAB_COLUMNS
        WHERE owner = :owner
          AND table_name = :table_name
        ORDER BY column_id
        """,
        owner=owner.upper(),
        table_name=table_name.upper(),
    )
    rows = cursor.fetchall()
    cursor.close()

    return [
        {
            "column_name": r[0],
            "data_type": r[1],
            "data_length": r[2],
            "data_precision": r[3],
            "data_scale": r[4],
            "nullable": r[5],
            "position": r[6],
        }
        for r in rows
    ]


def get_dependency_table_info(conn, dependencies: list) -> dict:
    """
    For each dependency found via get_dependencies(), fetches its column
    structure. Skips types that aren't tables/views (e.g. SYNONYM,
    PACKAGE) since ALL_TAB_COLUMNS won't have rows for those directly —
    synonyms would need resolving to their target first, noted as a gap.
    """
    table_info = {}
    for dep in dependencies:
        if dep["type"] not in ("TABLE", "VIEW"):
            continue
        key = f"{dep['owner']}.{dep['name']}"
        if key in table_info:
            continue  # already fetched
        columns = get_table_columns(conn, dep["owner"], dep["name"])
        table_info[key] = {
            "owner": dep["owner"],
            "table_name": dep["name"],
            "type": dep["type"],
            "column_count": len(columns),
            "columns": columns,
        }
    return table_info


def get_arguments(conn, package_name: str, database_name: str) -> list:
    """
    Pulls ALL_ARGUMENTS for every procedure/function in the package —
    parameter names, types, and IN/OUT direction.
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT object_name, argument_name, in_out, data_type, position
        FROM ALL_ARGUMENTS
        WHERE owner = :owner
          AND package_name = :pkg_name
        ORDER BY object_name, position
        """,
        owner=database_name.upper(),
        pkg_name=package_name.upper(),
    )
    rows = cursor.fetchall()
    cursor.close()

    args = {}
    for object_name, arg_name, in_out, data_type, position in rows:
        args.setdefault(object_name, []).append({
            "argument_name": arg_name,
            "in_out": in_out,
            "data_type": data_type,
            "position": position,
        })
    return args


def scan_db_links(source_text: str) -> list:
    """
    Scans the raw source text for '@LINK_NAME' patterns, which indicate
    a reference to a remote database via a database link. ALL_DEPENDENCIES
    doesn't always surface these cleanly, so a text scan is the reliable
    backstop.
    """
    matches = re.findall(r"@(\w+)", source_text)
    return sorted(set(m.upper() for m in matches))


def extract_package(package_name: str, database_name: str = "PQTEPF", force: bool = False) -> dict:
    """
    Runs the full Step 1 extraction for one package and saves the result
    to a local JSON file under extracted_packages/.

    If this package was already extracted before (its JSON file already
    exists), this SKIPS re-opening/re-fetching it from the database and
    just loads the existing file instead — no wasted DB calls, no
    duplicate work. Pass force=True to re-extract anyway (e.g. after
    the package source was changed in the database).
    """
    out_path = STORE_DIR / f"{package_name.upper()}.json"

    if out_path.exists() and not force:
        print(f"{package_name.upper()} already extracted — loading existing file, skipping DB fetch.")
        print(f"  ({out_path.resolve()})")
        with open(out_path, "r", encoding="utf-8") as f:
            return json.load(f)

    conn = get_db_connection(database_name)
    try:
        full_source = get_package_source(conn, package_name, database_name)
        procedures = split_into_procedures(full_source)
        dependencies = get_dependencies(conn, package_name, database_name)
        arguments = get_arguments(conn, package_name, database_name)
        table_info = get_dependency_table_info(conn, dependencies)

        # Attach per-procedure db-link flags, matched arguments, and tags
        for proc_name, proc_data in procedures.items():
            proc_data["db_links"] = scan_db_links(proc_data["source"])
            proc_data["arguments"] = arguments.get(proc_name, [])
            proc_data["procedure_label"] = parse_procedure_label(proc_data["source"])
            proc_data["column_tags"] = parse_column_tags(proc_data["source"])

        result = {
            "package_name": package_name.upper(),
            "database": database_name.upper(),
            "extracted_at": datetime.now().isoformat(),
            "procedure_count": len(procedures),
            "procedures": procedures,
            "package_level_dependencies": dependencies,
            "dependency_table_info": table_info,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(f"Extracted {len(procedures)} procedure(s) from {package_name.upper()}")
        print(f"Saved to: {out_path.resolve()}")

        return result
    finally:
        conn.close()


if __name__ == "__main__":
    # Example run — adjust the package name to test
    extract_package("PKG_BIEL_MBS", database_name="PQTEPF", force=True)