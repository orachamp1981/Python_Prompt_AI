"""Offline harness for the light path. No DB, no Ollama."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import light_path as lp

PKG = "PKG_BIEL_MBS"

with open(f"extracted_packages/{PKG}.json", "r", encoding="utf-8") as f:
    pkg = json.load(f)

proc_name = list(pkg["procedures"].keys())[0]
proc = pkg["procedures"][proc_name]

column_types = {}
for tbl in (pkg.get("dependency_table_info") or {}).values():
    tname = (tbl.get("table_name") or "").upper()
    for c in tbl.get("columns", []):
        column_types[f"{tname}.{c['column_name'].upper()}"] = c["data_type"]
print(f"COLUMN TYPES loaded: {len(column_types)}")

config = lp.resolve_config(proc["source"], PKG, column_types)
print("CONFIG:", None if not config else {
    "base_alias": config.base_alias,
    "count_key": config.count_key,
    "allowed": len(config.allowed_tables),
    "blocked": len(config.blocked_tables),
    "filters": config.mandatory_filters,
})

prov = lp.extract_provenance(proc["source"], config.base_alias)

label_index = lp.build_label_index(proc.get("column_tags", []))
known = set(prov.columns) | set(label_index.values())

print("\nALIAS MAP:")
for alias, entry in sorted(prov.alias_map.items()):
    print(f"  {alias:5} {entry['kind']:9} {entry['table'] or sorted(entry['tables'])}")

print(f"\nJOIN PREDICATES ({len(prov.join_predicates)}):")
for p in prov.join_predicates:
    print(f"  {p['raw']}")

print(f"\nBASE FILTERS ({len(prov.base_filters)}):")
for f_ in prov.base_filters:
    print(f"  {f_}")

print(f"\nCOLUMNS ({len(prov.columns)}):")
for name, col in sorted(prov.columns.items()):
    flag = ""
    if col["opaque_functions"]:
        flag = f" FUNC:{','.join(col['opaque_functions'])}"
    if col["is_subquery"]:
        flag += " SUBQ"
    verdict = "LIGHT" if all(config.table_allowed(t) for t in col["tables"]) and not col["opaque_functions"] else "FULL "
    print(f"  [{verdict}] {name:26} <- {','.join(col['tables']) or '(literal)'}{flag}")


# ------------------------------------------------------------------
# Scenario tests: resolved intents -> routing -> SQL
# ------------------------------------------------------------------

PARAMS = {"p_sdate": "2026-01-01", "p_edate": "2026-12-31",
          "p_company_code": 1, "date_explicit": True}

SCENARIOS = [
    # --- expected to route LIGHT ---
    ("total MBS policies (census)", {
        "measure": "COUNT", "measure_column": None,
        "dimensions": [], "filters": [], "date_column": None,
    }, True),
    ("policies issued in 2026", {
        "measure": "COUNT", "measure_column": None,
        "dimensions": [], "filters": [], "date_column": "ISSUANCE_DATE",
    }, True),
    ("policies submitted in 2026", {
        "measure": "COUNT", "measure_column": None,
        "dimensions": [], "filters": [], "date_column": "SUBMISSION_DATE",
    }, True),
    ("plan wise policy count", {
        "measure": "COUNT", "measure_column": None,
        "dimensions": ["PLAN_NAME"], "filters": [], "date_column": None,
    }, True),
    ("policy status wise count", {
        "measure": "COUNT", "measure_column": None,
        "dimensions": ["POLICY_STATUS"], "filters": [], "date_column": None,
    }, True),
    ("city wise policies", {
        "measure": "COUNT", "measure_column": None,
        "dimensions": ["CITY_NAME"], "filters": [], "date_column": None,
    }, True),
    ("occupation wise policies", {
        "measure": "COUNT", "measure_column": None,
        "dimensions": ["OCCUPATION"], "filters": [], "date_column": None,
    }, True),
    ("total contribution by plan", {
        "measure": "SUM", "measure_column": "CONTRIBUTION",
        "dimensions": ["PLAN_NAME"], "filters": [], "date_column": None,
    }, True),
    ("gender wise policies", {
        "measure": "COUNT", "measure_column": None,
        "dimensions": ["GENDER"], "filters": [], "date_column": None,
    }, True),
    ("in force policies", {
        "measure": "COUNT", "measure_column": None, "dimensions": [],
        "filters": [{"column": "POLICY_STATUS", "op": "=", "value": "IN FORCE"}],
        "date_column": None,
    }, True),
    ("product type wise policies", {
        "measure": "COUNT", "measure_column": None,
        "dimensions": ["PRODUCT_TYPE"], "filters": [], "date_column": None,
    }, True),

    # --- expected to fall back to the FULL procedure ---
    ("region wise business", {
        "measure": "SUM", "measure_column": "CONTRIBUTION",
        "dimensions": ["REGION"], "filters": [], "date_column": None,
    }, False),
    ("branch wise policies", {
        "measure": "COUNT", "measure_column": None,
        "dimensions": ["BRANCH_NAME"], "filters": [], "date_column": None,
    }, False),
    ("total received amount", {
        "measure": "SUM", "measure_column": "RECEIVED_AMOUNT",
        "dimensions": [], "filters": [], "date_column": None,
    }, False),
    ("total topup", {
        "measure": "SUM", "measure_column": "TOPUP",
        "dimensions": [], "filters": [], "date_column": None,
    }, False),
    ("age wise policies", {
        "measure": "COUNT", "measure_column": None,
        "dimensions": ["AGE"], "filters": [], "date_column": None,
    }, False),
    ("product name wise policies", {
        "measure": "COUNT", "measure_column": None,
        "dimensions": ["PRODUCT_NAME"], "filters": [], "date_column": None,
    }, False),
    ("agent wise policies", {
        "measure": "COUNT", "measure_column": None,
        "dimensions": ["AGENT_CODE"], "filters": [], "date_column": None,
    }, False),
    ("admin charges by plan", {
        "measure": "SUM", "measure_column": "ADMIN_ALLOCATION_CHARGES",
        "dimensions": ["PLAN_NAME"], "filters": [], "date_column": None,
    }, False),
]

print("\n" + "=" * 70)
print("  ROUTING SCENARIOS")
print("=" * 70)

failures = 0
for title, intent, expect_light in SCENARIOS:
    sql, decision = lp.build_light_sql(intent, prov, config, PARAMS, label_index, known)
    got_light = sql is not None
    ok = got_light == expect_light
    if not ok:
        failures += 1
    mark = "PASS" if ok else "FAIL"
    print(f"\n[{mark}] {title}")
    print(f"       route = {'LIGHT' if got_light else 'FULL'}")
    if not got_light:
        print(f"       why   = {decision.explain()}")
    else:
        for line in sql.splitlines():
            print(f"       | {line}")

print("\n" + "=" * 70)
print(f"  {len(SCENARIOS) - failures}/{len(SCENARIOS)} scenarios routed as expected")
print("=" * 70)


# ------------------------------------------------------------------
# Vocabulary resolution: business wording -> canonical column
# ------------------------------------------------------------------

TERMS = [
    "Policy Number", "policy", "Document No", "Client Name", "Policy Holder",
    "Customer", "Profession", "Nature of Job", "Date of Birth", "CNIC",
    "Date of Submission", "Policy Issuance Date", "Policy Started",
    "Plan Name", "Policy Plan", "Product Type", "Policy Status",
    "Contribution", "Payment Mode", "Amount Receivable", "Policy Topup",
    "Branch Name", "Region", "Region City", "Resident",
    "Participant Gender", "Participant Age", "Admin Charges",
    "Amount Received in Finance", "Policy Holder Country", "Invest_Status",
    "something made up",
]

print("\n" + "=" * 70)
print("  VOCABULARY RESOLUTION")
print("=" * 70)
for term in TERMS:
    hit = lp.resolve_term(term, label_index, known)
    reachable = "-"
    if hit and hit in prov.columns:
        col = prov.columns[hit]
        ok = all(config.table_allowed(t) for t in col["tables"]) and not col["opaque_functions"]
        reachable = "LIGHT" if ok else "FULL"
    elif hit:
        reachable = "not in base query"
    print(f"  {term:32} -> {str(hit):26} {reachable}")


# ------------------------------------------------------------------
# End-to-end: raw LLM-style intent -> resolved -> SQL
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("  END-TO-END INTENT RESOLUTION")
print("=" * 70)

RAW_INTENTS = [
    ("total policies", {"measure": "COUNT", "measure_column": None,
                        "dimensions": [], "filters": [], "date_column": None}),
    ("region wise business 2026", {"measure": "SUM", "measure_column": "Contribution",
                                   "dimensions": ["Region"], "filters": [],
                                   "date_column": "Policy Issuance Date"}),
    ("plan wise policy count", {"measure": "COUNT", "measure_column": None,
                                "dimensions": ["Policy Plan"], "filters": [],
                                "date_column": None}),
    ("male clients in Karachi", {"measure": "COUNT", "measure_column": None,
                                 "dimensions": [],
                                 "filters": [{"column": "Participant Gender", "op": "=", "value": "Male"},
                                             {"column": "Region City", "op": "=", "value": "Karachi"}],
                                 "date_column": None}),
    # LIST must project, not aggregate
    ("list policies with client and plan", {"measure": "LIST", "measure_column": None,
                                            "dimensions": ["Policy Number", "Client Name", "Policy Plan"],
                                            "filters": [], "date_column": None}),
    # A term the vocabulary does not know must NOT silently produce a census
    ("hallucinated dimension", {"measure": "COUNT", "measure_column": None,
                                "dimensions": ["PROFIT_SHARE"], "filters": [],
                                "date_column": None}),
]

for title, raw in RAW_INTENTS:
    sql, decision = lp.build_light_sql(raw, prov, config, PARAMS, label_index, known)
    print(f"\n  prompt : {title}")
    print(f"  intent : {json.dumps(raw)}")
    print(f"  resolved: {json.dumps(decision.intent)}")
    print(f"  route  : {'LIGHT' if sql else 'FULL'}")
    if sql:
        for line in sql.splitlines():
            print(f"      | {line}")
    else:
        print(f"      why: {decision.explain()}")
    for c in decision.cautions:
        print(f"      ! {c}")

