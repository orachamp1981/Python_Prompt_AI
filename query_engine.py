"""
BIEL Query Engine — Prompt in, SQL out.

Reads extracted package JSON (from Step 1), builds a focused context
from @labels and procedure parameters, sends to local Ollama, and
returns Oracle SQL for user validation before execution.

No cloud APIs. No tokenization. Local only.

Usage:
    python query_engine.py
    python query_engine.py --package PKG_BIEL_MBS
    python query_engine.py --package PKG_BIEL_MBS --company 1
"""

import json
import argparse
import requests
from pathlib import Path
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from databases import get_db_connection, execute_pqtepf_query
import light_path as lp

# ============================================================
# CONFIGURATION
# ============================================================

STORE_DIR = Path("extracted_packages")
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1"
DEFAULT_COMPANY_CODE = 1
DEFAULT_DATABASE = "PQTEPF"
ALL_TIME_START = "1900-01-01"


# ============================================================
# CONTEXT BUILDER
# ============================================================

def load_package(package_name: str) -> dict:
    path = STORE_DIR / f"{package_name.upper()}.json"
    if not path.exists():
        available = [f.stem for f in STORE_DIR.glob("*.json")]
        raise FileNotFoundError(
            f"Package '{package_name}' not extracted yet.\n"
            f"Available: {available}\n"
            f"Run: python extract_package_step1.py"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_context(package_data: dict) -> str:
    """
    Builds the LLM context from extracted @labels and parameters.
    Only sends the business vocabulary — NOT the inner SQL.
    """
    lines = []
    pkg_name = package_data["package_name"]
    database = package_data["database"]

    lines.append(f"Package: {pkg_name}")
    lines.append(f"Database: {database}")
    lines.append("")

    for proc_name, proc_data in package_data["procedures"].items():
        proc_label = proc_data.get("procedure_label") or proc_name
        lines.append(f"Procedure: {pkg_name}.{proc_name}")
        lines.append(f"Business: {proc_label}")
        lines.append("")

        # Parameters
        lines.append("Parameters (IN):")
        for arg in proc_data.get("arguments", []):
            if arg["in_out"] == "IN":
                lines.append(f"  - {arg['argument_name']} ({arg['data_type']})")
        lines.append("")

        # Available columns with business labels
        lines.append("Available columns:")
        PLSQL_KEYWORDS = {"BEGIN", "END", "IS", "AS", "DECLARE", "OPEN", "CLOSE", "FETCH"}
        for tag in proc_data.get("column_tags", []):
            if tag["column"] in PLSQL_KEYWORDS:
                continue
            if tag["tag_type"] == "label":
                col = tag["column"]
                labels = " | ".join(tag["labels"])
                lines.append(f"  - {col}: {labels}")
            elif tag["tag_type"] == "derived_concept":
                col = tag["column"]
                logic = tag.get("logic", "")
                lines.append(f"  - {col} [DERIVED]: {logic}")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# PROMPT BUILDER
# ============================================================

SYSTEM_PROMPT_TEMPLATE = """You are an Oracle SQL query generator for a business intelligence engine.

You generate SQL queries that wrap stored procedure calls. The procedure returns
a REF CURSOR with known columns. You ONLY use columns listed below.

{context}

RULES:
1. Always call the procedure using this pattern:
   SELECT <columns/aggregations>
   FROM TABLE({package}.{procedure}(
       TO_DATE(:p_sdate, 'YYYY-MM-DD'),
       TO_DATE(:p_edate, 'YYYY-MM-DD'),
       :p_company_code
   ))
2. Only use columns listed in "Available columns" above.
3. For "wise", "by", "per", "breakdown" -> use GROUP BY.
4. For "how many", "count" -> use COUNT(*).
5. For "total", "sum", "business" (when referring to amounts) -> use SUM(CONTRIBUTION).
6. For date references like "2026", "last year", "January" -> set :p_sdate and :p_edate values.
7. If user mentions a specific column filter (e.g., "in force", "Punjab"), add WHERE clause.
8. Always ORDER BY the result when using GROUP BY.
9. Output ONLY the SQL query. No explanation, no markdown, no comments.
10. Use Oracle SQL syntax only (TO_DATE, NVL, DECODE, etc.).
11. CRITICAL: When using GROUP BY, ALWAYS include the grouped column in the SELECT list.
    Example: SELECT REGION, COUNT(*) AS POLICY_COUNT FROM ... GROUP BY REGION
12. The procedure call syntax for Oracle pipelined/REF cursor wrapper is:
    SELECT ... FROM TABLE({package}.{procedure}(...))
    Use this exact pattern. The engine will handle the actual execution mechanism.
13. Always alias aggregate columns (AS TOTAL_CONTRIBUTION, AS POLICY_COUNT, etc.).
14. When user says "business" in context of amounts/money, use SUM(CONTRIBUTION).
    When user says "business" in context of counting, use COUNT(DOCUMENT_CODE).
    If ambiguous, provide BOTH: COUNT and SUM.
15. IMPORTANT — entity vs amount distinction:
    - "policies", "clients", "participants", "cases" = COUNTABLE entities → use COUNT(DOCUMENT_CODE) or COUNT(*)
    - "business", "contribution", "premium", "amount", "total money" = AMOUNTS → use SUM(CONTRIBUTION)
    - "total policies" = COUNT(DOCUMENT_CODE), NOT SUM
    - "total business" = SUM(CONTRIBUTION), NOT COUNT
    - When in doubt about "total X": if X is a thing you can count (policy, client, branch), use COUNT.
      If X is money/amount (business, contribution, premium), use SUM.
"""


def build_system_prompt(package_data: dict) -> str:
    context = build_context(package_data)
    pkg_name = package_data["package_name"]
    proc_names = list(package_data["procedures"].keys())
    procedure = proc_names[0] if proc_names else "UNKNOWN"

    return SYSTEM_PROMPT_TEMPLATE.format(
        context=context,
        package=pkg_name,
        procedure=procedure,
    )


# ============================================================
# OLLAMA CALL
# ============================================================

def call_ollama(system_prompt: str, user_prompt: str) -> str:
    full_prompt = f"{system_prompt}\n\nUser question: {user_prompt}\n\nSQL:"

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 1024,
                },
            },
            timeout=300,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Cannot reach Ollama at {OLLAMA_URL}\n"
            "Is Ollama running? Start it with: ollama serve"
        )

    raw = response.json().get("response", "").strip()
    # Clean markdown fences if model adds them
    raw = raw.replace("```sql", "").replace("```", "").strip()
    # Take only the first SQL statement
    if ";" in raw:
        raw = raw[:raw.index(";") + 1]
    return raw


# ============================================================
# INTENT EXTRACTION (drives the light path)
# ============================================================

INTENT_PROMPT_TEMPLATE = """You resolve a business question into a structured query intent.

{context}

Return ONLY one JSON object. No markdown, no explanation, no trailing text.
Shape:
{{"measure":"COUNT|SUM|AVG|MIN|MAX|LIST","measure_column":"<column or null>","dimensions":["<column>"],"filters":[{{"column":"<column>","op":"=|<>|>|<|>=|<=|LIKE|IN","value":"<literal>"}}],"date_column":"<column or null>"}}

RULES:
1. Use ONLY the exact column names listed above — the identifier before the colon.
2. Countable things ("policies", "clients", "participants", "cases") -> measure COUNT, measure_column null.
3. Amounts ("business", "contribution", "premium", "amount", "topup", "due") -> measure SUM and set measure_column to that amount column.
4. "total policies" is COUNT. "total business" is SUM of CONTRIBUTION.
5. "wise", "by", "per", "breakdown" -> put the grouping column in dimensions.
6. Add a filter only when the user names a concrete value ("in force", "male", "individual", "Karachi").
7. date_column is chosen by the user's wording, never by preference:
   - "submitted", "submission", "entered", "applied" -> SUBMISSION_DATE
   - "issued", "issuance", "approved", "started", "commenced", "confirmed" -> ISSUANCE_DATE
   - question mentions no time period at all -> null
8. Never invent a column. Use null or an empty list when nothing matches.
"""


def build_intent_prompt(package_data: dict) -> str:
    return INTENT_PROMPT_TEMPLATE.format(context=build_context(package_data))


def parse_json_object(raw: str) -> dict:
    """Pulls the first JSON object out of a model response."""
    text = (raw or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"No JSON object in model response: {raw[:200]}")

    return json.loads(text[start:end + 1])


def call_ollama_intent(intent_prompt: str, user_prompt: str) -> dict:
    full_prompt = f"{intent_prompt}\n\nUser question: {user_prompt}\n\nJSON:"

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": full_prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0, "num_predict": 512},
            },
            timeout=300,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Cannot reach Ollama at {OLLAMA_URL}\n"
            "Is Ollama running? Start it with: ollama serve"
        )

    return parse_json_object(response.json().get("response", ""))


# ============================================================
# SQL POST-PROCESSING
# ============================================================

def extract_inner_sql(package_data: dict) -> str:
    """
    Extracts the core SELECT statement from the procedure source.
    This is the SQL inside OPEN p_result FOR ... ;
    """
    import re
    proc_names = list(package_data["procedures"].keys())
    if not proc_names:
        raise ValueError("No procedures found in package")

    source = package_data["procedures"][proc_names[0]]["source"]

    # Find OPEN p_result FOR ... pattern
    match = re.search(
        r"OPEN\s+\w+\s+FOR\s*\n?(.*?)(?:;\s*END|\s*END\s+\w+\s*;)",
        source,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        sql = match.group(1).strip().rstrip(";").strip()
        return sql

    # Fallback: find first SELECT ... to last semicolon before END
    match = re.search(
        r"(SELECT\b.*?)(?:;\s*$|;\s*END)",
        source,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip().rstrip(";").strip()

    raise ValueError("Could not extract inner SQL from procedure source")


def resolve_procedure_call(sql: str, package_data: dict, params: dict) -> str:
    """
    Replaces PROCEDURE_OUTPUT or TABLE(pkg.proc(...)) patterns with the
    actual inner SQL from the procedure, with bind parameters substituted.
    """
    import re

    sdate = params.get("p_sdate", "2026-01-01")
    edate = params.get("p_edate", "2026-12-31")
    company = params.get("p_company_code", DEFAULT_COMPANY_CODE)

    pkg_name = package_data["package_name"]
    proc_names = list(package_data["procedures"].keys())
    proc_name = proc_names[0] if proc_names else "UNKNOWN"

    # Get the inner SQL from the procedure
    inner_sql = extract_inner_sql(package_data)

    # Replace PL/SQL bind variables with literal values in the inner SQL
    inner_sql = re.sub(
        r"\bp_?sdate\b",
        f"TO_DATE('{sdate}', 'YYYY-MM-DD')",
        inner_sql,
        flags=re.IGNORECASE,
    )
    inner_sql = re.sub(
        r"\bp_?edate\b",
        f"TO_DATE('{edate}', 'YYYY-MM-DD')",
        inner_sql,
        flags=re.IGNORECASE,
    )
    inner_sql = re.sub(
        r"\bp_?company_?code\b",
        str(company),
        inner_sql,
        flags=re.IGNORECASE,
    )

    # Pattern 1: PROCEDURE_OUTPUT placeholder
    if "PROCEDURE_OUTPUT" in sql.upper():
        idx = sql.upper().index("PROCEDURE_OUTPUT")
        sql = sql[:idx] + f"(\n{inner_sql}\n)" + sql[idx + len("PROCEDURE_OUTPUT"):]
        return sql

    # Pattern 2: TABLE(PKG_BIEL_MBS.MBS_DOCUMENT_DETAIL(...))
    table_pattern = re.compile(
        rf"TABLE\s*\(\s*{re.escape(pkg_name)}\s*\.\s*\w+\s*\(.*?\)\s*\)",
        re.IGNORECASE | re.DOTALL,
    )
    if table_pattern.search(sql):
        sql = table_pattern.sub(f"(\n{inner_sql}\n)", sql)
        return sql

    # Pattern 3: Direct procedure call without TABLE() wrapper
    proc_pattern = re.compile(
        rf"{re.escape(pkg_name)}\s*\.\s*{re.escape(proc_name)}\s*\(.*?\)",
        re.IGNORECASE | re.DOTALL,
    )
    if proc_pattern.search(sql):
        sql = proc_pattern.sub(f"(\n{inner_sql}\n)", sql)
        return sql

    # Pattern 4: Any remaining :p_company_code or bind vars in outer SQL
    sql = sql.replace(":p_company_code", str(company))
    sql = sql.replace(":p_sdate", f"TO_DATE('{sdate}', 'YYYY-MM-DD')")
    sql = sql.replace(":p_edate", f"TO_DATE('{edate}', 'YYYY-MM-DD')")

    return sql


def extract_dates_from_prompt(prompt: str) -> dict:
    """Extract date parameters from natural language prompt."""
    import re

    params = {}
    current_year = date.today().year

    # Year detection: "2026", "for 2025"
    year_match = re.search(r"\b(20\d{2})\b", prompt)
    if year_match:
        year = int(year_match.group(1))
        params["p_sdate"] = f"{year}-01-01"
        params["p_edate"] = f"{year}-12-31"

    # Month detection: "January 2026", "Jan 2026"
    months = {
        "january": 1, "jan": 1, "february": 2, "feb": 2,
        "march": 3, "mar": 3, "april": 4, "apr": 4,
        "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
        "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
        "october": 10, "oct": 10, "november": 11, "nov": 11,
        "december": 12, "dec": 12,
    }

    for month_name, month_num in months.items():
        pattern = rf"\b{month_name}\s*(20\d{{2}})?\b"
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            year = int(match.group(1)) if match.group(1) else current_year
            params["p_sdate"] = f"{year}-{month_num:02d}-01"
            if month_num == 12:
                params["p_edate"] = f"{year + 1}-01-01"
            else:
                params["p_edate"] = f"{year}-{month_num + 1:02d}-01"
            break

    # "last year"
    if "last year" in prompt.lower():
        params["p_sdate"] = f"{current_year - 1}-01-01"
        params["p_edate"] = f"{current_year - 1}-12-31"

    # "this year" / "current year"
    if "this year" in prompt.lower() or "current year" in prompt.lower():
        params["p_sdate"] = f"{current_year}-01-01"
        params["p_edate"] = f"{current_year}-12-31"

    # Default if nothing detected.
    #
    # A question with no time period ("total policies") means all policies
    # ever issued, including ones written in 2000 that are still running.
    # Defaulting to the current year would silently under-report, so the
    # engine widens the range and flags that the prompt carried no date.
    if "p_sdate" not in params:
        params["p_sdate"] = ALL_TIME_START
        params["p_edate"] = f"{current_year + 1}-12-31"
        params["date_explicit"] = False
    else:
        params["date_explicit"] = True

    return params


# ============================================================
# LIGHT PATH PREPARATION
# ============================================================

def prepare_light_path(package_data: dict) -> Optional[dict]:
    """
    Parses the procedure once and derives column provenance plus the
    DBA-declared whitelist. Returns None when the package has not opted
    in, in which case only the full path is available.
    """
    proc_names = list(package_data["procedures"].keys())
    if not proc_names:
        return None

    proc = package_data["procedures"][proc_names[0]]
    package_name = package_data["package_name"]

    # Oracle metadata for the package's dependencies, used to prove that a
    # wrapped column really is a DATE before simplifying a date filter.
    column_types = {}
    for table_info in (package_data.get("dependency_table_info") or {}).values():
        table_name = (table_info.get("table_name") or "").upper()
        if not table_name:
            continue
        for column in table_info.get("columns", []):
            column_types[f"{table_name}.{column['column_name'].upper()}"] = column["data_type"]

    config = lp.resolve_config(proc["source"], package_name, column_types)
    if config is None:
        return None

    prov = lp.extract_provenance(proc["source"], config.base_alias)
    if prov is None:
        return None

    label_index = lp.build_label_index(proc.get("column_tags", []))
    known_columns = set(prov.columns) | set(label_index.values())

    return {
        "config": config,
        "prov": prov,
        "label_index": label_index,
        "known_columns": known_columns,
    }


# ============================================================
# DUAL-PATH PLANNING
# ============================================================

def plan_query(prompt: str, package_data: dict, system_prompt: str,
               intent_prompt: str, light_ctx: Optional[dict], params: dict) -> dict:
    """
    Runs both generators concurrently and returns whatever each produced.

    The full path is the existing behaviour: Ollama writes SQL around the
    procedure. The light path asks Ollama only for a structured intent,
    then Python resolves it against provenance and renders SQL from a
    template — so joins and tables are never model decisions.
    """
    with ThreadPoolExecutor(max_workers=2) as pool:
        full_future = pool.submit(call_ollama, system_prompt, prompt)
        intent_future = (
            pool.submit(call_ollama_intent, intent_prompt, prompt)
            if light_ctx else None
        )

        plan = {
            "params": params,
            "full_sql": None,
            "full_error": None,
            "intent": None,
            "intent_error": None,
            "light_sql": None,
            "decision": None,
        }

        try:
            raw_sql = full_future.result()
            plan["full_sql"] = resolve_procedure_call(raw_sql, package_data, params)
        except Exception as exc:
            plan["full_error"] = str(exc)

        if intent_future is not None:
            try:
                intent = intent_future.result()
                plan["intent"] = intent
                sql, decision = lp.build_light_sql(
                    intent,
                    light_ctx["prov"],
                    light_ctx["config"],
                    params,
                    light_ctx["label_index"],
                    light_ctx["known_columns"],
                )
                plan["light_sql"] = sql
                plan["decision"] = decision
            except Exception as exc:
                plan["intent_error"] = str(exc)

    return plan


# ============================================================
# EXECUTION
# ============================================================

def execute_query(sql: str, database: str = DEFAULT_DATABASE):
    """Execute the generated SQL and return results as DataFrame."""
    import pandas as pd
    conn = get_db_connection(database)
    try:
        df = pd.read_sql(sql, con=conn)
        return df
    finally:
        conn.close()


def render_plan(plan: dict, full_preview_lines: int = 25) -> None:
    """Prints both candidate queries so they can be compared before running."""
    bar = "-" * 66

    if plan.get("intent"):
        print("\n" + bar)
        print("  INTENT")
        print(f"    model said : {json.dumps(plan['intent'])}")
        decision = plan.get("decision")
        if decision is not None and decision.intent:
            print(f"    resolved to: {json.dumps(decision.intent)}")

    print("\n" + bar)
    print("  [1] LIGHT PATH")
    if plan.get("light_sql"):
        for line in plan["light_sql"].splitlines():
            print(f"      {line}")
        decision = plan.get("decision")
        if decision and decision.cautions:
            print("      cautions:")
            for caution in decision.cautions:
                print(f"        ! {caution}")
    elif plan.get("decision") is not None:
        print(f"      not available — {plan['decision'].explain()}")
    elif plan.get("intent_error"):
        print(f"      intent error — {plan['intent_error']}")
    else:
        print("      not configured for this package")

    print(bar)
    full_sql = plan.get("full_sql")
    print("  [2] FULL PATH — complete procedure SQL")
    if full_sql:
        lines = full_sql.splitlines()
        for line in lines[:full_preview_lines]:
            print(f"      {line}")
        if len(lines) > full_preview_lines:
            print(f"      ... {len(lines) - full_preview_lines} more lines (press V to view)")
    else:
        print(f"      error — {plan.get('full_error')}")
    print(bar)


# ============================================================
# INTERACTIVE ENGINE
# ============================================================

def run_engine(package_name: str, company_code: int = DEFAULT_COMPANY_CODE):
    print("=" * 60)
    print("  BIEL QUERY ENGINE — Prompt to SQL")
    print("=" * 60)

    # Load extracted package
    package_data = load_package(package_name)
    print(f"\nPackage loaded: {package_data['package_name']}")
    print(f"Procedures: {list(package_data['procedures'].keys())}")
    print(f"Company code: {company_code}")

    # Build system prompt
    system_prompt = build_system_prompt(package_data)
    intent_prompt = build_intent_prompt(package_data)

    light_ctx = prepare_light_path(package_data)
    if light_ctx:
        cfg = light_ctx["config"]
        light_cols = sum(
            1 for c in light_ctx["prov"].columns.values()
            if all(cfg.table_allowed(t) for t in c["tables"]) and not c["opaque_functions"]
        )
        print(f"Light path: enabled (base {cfg.base_table}, "
              f"{light_cols}/{len(light_ctx['prov'].columns)} columns reachable)")
    else:
        print("Light path: not configured — full procedure only")

    print(f"\nOllama model: {OLLAMA_MODEL}")
    print(f"Ollama URL: {OLLAMA_URL}")
    print("\n" + "=" * 60)
    print("  Type your business question.")
    print("  Commands: 'exit' to quit, 'context' to see loaded labels")
    print("=" * 60)

    while True:
        try:
            prompt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not prompt:
            continue

        if prompt.lower() in ("exit", "quit", "q"):
            print("Bye.")
            break

        if prompt.lower() == "context":
            print("\n" + build_context(package_data))
            continue

        params = extract_dates_from_prompt(prompt)
        params["p_company_code"] = company_code

        if params.get("date_explicit"):
            print(f"\n  Period: {params['p_sdate']} to {params['p_edate']}")
        else:
            print(f"\n  Period: none in prompt -> all-time "
                  f"({params['p_sdate']} to {params['p_edate']})")
        print(f"  Company: {params['p_company_code']}")
        print("  Running light + full path in parallel...", end="", flush=True)

        try:
            plan = plan_query(
                prompt, package_data, system_prompt,
                intent_prompt, light_ctx, params
            )
        except ConnectionError as e:
            print(f"\n\n  ERROR: {e}")
            continue
        except Exception as e:
            print(f"\n\n  ERROR: {e}")
            continue

        print(" done.")
        render_plan(plan)

        while True:
            action = input("\n  [1] run LIGHT / [2] run FULL / "
                           "[V]iew full SQL / [R]egenerate / [S]kip > ").strip().lower()

            if action == "v":
                print("\n" + "-" * 60)
                print(plan.get("full_sql") or "(no full SQL)")
                print("-" * 60)
                continue

            if action == "1":
                target, label = plan.get("light_sql"), "LIGHT"
            elif action == "2":
                target, label = plan.get("full_sql"), "FULL"
            elif action == "r":
                hint = input("  Correction hint (optional): ").strip()
                retry = f"{prompt}\n\nIMPORTANT CORRECTION: {hint}" if hint else prompt
                try:
                    plan = plan_query(
                        retry, package_data, system_prompt,
                        intent_prompt, light_ctx, params
                    )
                    render_plan(plan)
                except Exception as e:
                    print(f"\n  ERROR: {e}")
                continue
            else:
                print("  Skipped.")
                break

            if not target:
                print("  That query is not available.")
                continue

            print(f"\n  Executing {label} path...")
            try:
                df = execute_query(target, package_data["database"])
                print(f"\n  Rows returned: {len(df)}")
                print(f"  Columns: {list(df.columns)}")
                print("\n" + df.head(20).to_string(index=False))
                if len(df) > 20:
                    print(f"\n  ... {len(df) - 20} more rows")
            except Exception as e:
                print(f"\n  EXECUTION ERROR: {e}")
            break


# ============================================================
# SINGLE QUERY MODE (non-interactive)
# ============================================================

def plan(prompt: str, package_name: str = "PKG_BIEL_MBS",
         company_code: int = DEFAULT_COMPANY_CODE) -> dict:
    """
    Programmatic API — returns both candidate queries without executing.
    Keys: light_sql, full_sql, decision, intent, params.
    """
    package_data = load_package(package_name)
    system_prompt = build_system_prompt(package_data)
    intent_prompt = build_intent_prompt(package_data)
    light_ctx = prepare_light_path(package_data)

    params = extract_dates_from_prompt(prompt)
    params["p_company_code"] = company_code

    return plan_query(prompt, package_data, system_prompt, intent_prompt, light_ctx, params)


def generate_sql(prompt: str, package_name: str = "PKG_BIEL_MBS",
                 company_code: int = DEFAULT_COMPANY_CODE) -> str:
    """Returns the full-path SQL, preserving the original single-query API."""
    return plan(prompt, package_name, company_code)["full_sql"]


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BIEL Query Engine")
    parser.add_argument("--package", default="PKG_BIEL_MBS", help="Package name")
    parser.add_argument("--company", type=int, default=DEFAULT_COMPANY_CODE, help="Company code")
    parser.add_argument("--model", default=OLLAMA_MODEL, help="Ollama model name")
    parser.add_argument("--prompt", default=None, help="Single prompt (non-interactive)")
    parser.add_argument("--path", choices=["light", "full", "both"], default="both",
                        help="Which query to print in single-prompt mode")
    args = parser.parse_args()

    OLLAMA_MODEL = args.model

    if args.prompt:
        result = plan(args.prompt, args.package, args.company)
        render_plan(result, full_preview_lines=10 ** 6)
        if args.path != "both":
            print(result.get(f"{args.path}_sql") or f"-- {args.path} path unavailable")
    else:
        run_engine(args.package, args.company)
