"""Tests for package-driven prompt interpretation and SQL composition."""

import json
from io import BytesIO
import unittest

from package_metadata import PackageMetadata
from package_metadata import _read_annotations
from package_metadata import _read_business_concepts
from package_metadata import _read_direct_concepts
from package_metadata import _read_lookup_concepts
from prompt_engine import _build_sql
from prompt_engine import _cash_value_lookup
from prompt_engine import _distinct_dimension_count_plan
from prompt_engine import _parse_request_date_range
from prompt_engine import _read_plan
from prompt_engine import _resolve_annotated_measure
from prompt_engine import _resolve_distinct_dimension_count
from prompt_engine import _resolve_output_columns
from prompt_engine import _apply_seeded_semantics
from prompt_engine import _requested_output_aliases
from prompt_engine import _seeded_direct_count_plan
from prompt_engine import _seeded_lookup_plan
from prompt_engine import interpret_prompt
from unittest.mock import patch


def sample_metadata():
    source = {
        "PACKAGE": """
PROCEDURE MBS_DOCUMENT_DETAIL (
  P_SDATE IN DATE,
  P_EDATE IN DATE,
  P_COMPANY_CODE IN NUMBER,
  P_RESULT OUT SYS_REFCURSOR
);
""",
        "PACKAGE BODY": """
PROCEDURE MBS_DOCUMENT_DETAIL (
  P_SDATE IN DATE,
  P_EDATE IN DATE,
  P_COMPANY_CODE IN NUMBER,
  P_RESULT OUT SYS_REFCURSOR
) IS
BEGIN
  -- @business: MBS
  -- @capability: policy_details
  OPEN P_RESULT FOR
    SELECT REGION, RECEIVED_AMOUNT
    FROM BUSINESS_010226
    WHERE TRANS_DATE BETWEEN P_SDATE AND P_EDATE
      AND COMPANYID_FK = P_COMPANY_CODE;
END MBS_DOCUMENT_DETAIL;
"""
    }
    return PackageMetadata(
        owner="PQTEPF",
        package="PKG_BIEL_MBS",
        source=source,
        dependencies=[
            {"owner": "PQTEPF", "name": "BUSINESS_010226", "type": "TABLE", "link": ""}
        ],
        annotations=[
            {
                "procedure": "MBS_DOCUMENT_DETAIL",
                "line": "1",
                "kind": "label",
                "value": "Region | Regional",
                "field": "REGION",
            },
            {
                "procedure": "MBS_DOCUMENT_DETAIL",
                "line": "2",
                "kind": "label",
                "value": "Received Amount | Contribution",
                "field": "RECEIVED_AMOUNT",
            },
        ],
    )


def seeded_metadata():
    metadata = sample_metadata()
    metadata.source["PACKAGE BODY"] = metadata.source["PACKAGE BODY"].replace(
        "SELECT REGION, RECEIVED_AMOUNT",
        "SELECT REGION, RECEIVED_AMOUNT, DOCUMENT_CODE, SUBMISSION_DATE",
    )
    metadata.annotations.extend(
        [
            {
                "procedure": "MBS_DOCUMENT_DETAIL",
                "line": "3",
                "kind": "label",
                "value": "Policy Number | Policy | Document Code",
                "field": "DOCUMENT_CODE",
            },
            {
                "procedure": "MBS_DOCUMENT_DETAIL",
                "line": "4",
                "kind": "label",
                "value": "Submission Date | Date of Submission",
                "field": "SUBMISSION_DATE",
            },
        ]
    )
    metadata.business_concepts = _read_business_concepts(
        metadata.owner,
        metadata.package,
    )
    metadata.direct_concepts = _read_direct_concepts(
        metadata.owner,
        metadata.package,
    )
    metadata.lookup_concepts = _read_lookup_concepts(
        metadata.owner,
        metadata.package,
    )
    return metadata


class PackageAnnotationTests(unittest.TestCase):
    def test_annotations_keep_procedure_and_field(self):
        annotations = _read_annotations(
            "PROCEDURE MBS_DETAIL IS\n"
            "BEGIN\n"
            "  DOCUMENT_CODE, -- @label: Policy Number | Policy\n"
            "  -- @business: MBS\n"
            "END;\n"
        )

        self.assertEqual(
            [(item["procedure"], item["kind"], item.get("field")) for item in annotations],
            [
                ("MBS_DETAIL", "label", "DOCUMENT_CODE"),
                ("MBS_DETAIL", "business", None),
            ],
        )

    def test_field_catalog_links_labels_to_canonical_columns(self):
        fields = sample_metadata().field_catalog()

        self.assertIn("Region", fields["REGION"]["labels"])
        self.assertIn("Contribution", fields["RECEIVED_AMOUNT"]["labels"])
        self.assertEqual(fields["REGION"]["lineage"], [])

    def test_field_catalog_ignores_annotated_columns_not_returned_by_cursor(self):
        metadata = sample_metadata()
        metadata.source["PACKAGE BODY"] = metadata.source["PACKAGE BODY"].replace(
            "SELECT REGION, RECEIVED_AMOUNT",
            "SELECT REGION, RECEIVED_AMOUNT\n      -- NVL(TOPUP, 0) TOPUP,",
        )
        metadata.annotations.append(
            {
                "procedure": "MBS_DOCUMENT_DETAIL",
                "line": "3",
                "kind": "label",
                "value": "Topup",
                "field": "TOPUP",
            }
        )

        self.assertNotIn("TOPUP", metadata.field_catalog())

    def test_seeded_concept_bundle_targets_only_live_cursor_fields(self):
        metadata = seeded_metadata()
        fields = metadata.field_catalog()

        self.assertEqual(
            {concept["id"] for concept in metadata.business_concepts},
            {
                "business_amount",
                "case_count",
                "policy_count",
                "policy_registration_date",
            },
        )
        self.assertIn("RECEIVED_AMOUNT", fields)
        self.assertNotIn("TOPUP", fields)
        self.assertIn("SEEDED BUSINESS CONCEPTS", metadata.prompt_context("Top up"))
        context = metadata.prompt_context("Describe the available information.")
        self.assertIn("ADDITIONAL APPROVED OPERATIONS", context)
        self.assertIn("policy_count -> COUNT(EPF_UW_DOCUMENT.DOCUMENT_ID_PK)", context)
        self.assertIn("policies_by_cnic", context)

    def test_business_concept_is_resolved_from_live_field_annotation(self):
        metadata = sample_metadata()
        metadata.annotations.append(
            {
                "procedure": "MBS_DOCUMENT_DETAIL",
                "line": "3",
                "kind": "concept",
                "value": "Business | Contribution | Topup | Received Amount",
                "field": "RECEIVED_AMOUNT",
            }
        )

        context = metadata.prompt_context("Region-wise business, August 2026")

        self.assertIn("RECEIVED_AMOUNT", context)
        self.assertIn("Business | Contribution | Topup | Received Amount", context)

    def test_plural_prompt_matches_singular_annotated_field(self):
        context = sample_metadata().prompt_context(
            "How many regions are there in MBS system"
        )

        self.assertIn("REGION", context)

class PromptPlanningTests(unittest.TestCase):
    def test_semantic_information_intent_returns_dynamic_metadata_capabilities(self):
        metadata = seeded_metadata()
        classifier_result = json.dumps({"intent": "information"})
        response = BytesIO(
            json.dumps(
                {"message": {"content": classifier_result}}
            ).encode("utf-8")
        )
        prompt = "Explain the kinds of information this workspace can return."

        with patch("prompt_engine.urlopen", return_value=response) as ollama:
            result = interpret_prompt(prompt, metadata)

        self.assertEqual(result["status"], "informational")
        self.assertEqual(result["message"], metadata.describe_capabilities())
        self.assertIn("business", result["message"])
        self.assertIn("case", result["message"])
        self.assertIn("total number of policies", result["message"])
        self.assertIn("policy number by CNIC", result["message"])
        self.assertIsNone(result["sql"])
        self.assertEqual(result["parameters"], {})
        request = ollama.call_args.args[0]
        request_payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request_payload["model"], "qwen2.5:1.5b")
        self.assertIn("Classify the user's intent", request_payload["messages"][0]["content"])
        self.assertEqual(request_payload["messages"][1]["content"], prompt)
        ollama.assert_called_once()

    def test_total_policy_count_uses_direct_document_primary_key_count(self):
        metadata = seeded_metadata()
        prompts = [
            "Let me have the total number of policies",
            "How many policies are there?",
        ]
        for prompt in prompts:
            with patch(
                "prompt_engine.urlopen",
                side_effect=AssertionError("A seeded policy count must not call Ollama."),
            ):
                result = interpret_prompt(prompt, metadata)

            self.assertIsNotNone(result)
            self.assertEqual(result["sql"], (
                'SELECT COUNT(DOCUMENT_ID_PK) AS "Total Number of Policies"\n'
                "FROM EPF_UW_DOCUMENT"
            ))
            self.assertEqual(result["operation"], "direct_count")

    def test_policy_lookup_by_cnic_uses_seeded_join_and_bound_value(self):
        metadata = seeded_metadata()
        with patch(
            "prompt_engine.urlopen",
            side_effect=AssertionError("A seeded CNIC lookup must not call Ollama."),
        ):
            result = interpret_prompt(
                "Let me know the policies with CNIC 4220165562223?",
                metadata,
            )

        self.assertEqual(result["status"], "ready")
        self.assertEqual(
            result["sql"],
            'SELECT DISTINCT T1.DOCUMENT_CODE AS "Policy Number"\n'
            "FROM EPF_UW_DOCUMENT T1\n"
            "JOIN EPF_UW_EMPLOYEES T2 "
            "ON T1.DOCUMENT_ID_PK = T2.DOCUMENT_ID_FK\n"
            "WHERE T2.CNIC_NO = :P_CNIC "
            "AND T1.COMPANYID_FK = :P_COMPANY_CODE",
        )
        self.assertEqual(result["parameters"], {"P_CNIC": "4220165562223"})
        self.assertEqual(result["empty_result_message"], "Policy does not exist")

    def test_policy_lookup_requires_one_valid_cnic(self):
        metadata = seeded_metadata()
        result = _seeded_lookup_plan("Let me know the policies with CNIC 123", metadata)

        self.assertEqual(result["status"], "clarify")
        self.assertIsNone(result["sql"])
        self.assertIn("valid CNIC", result["message"])

    def test_resolves_model_count_distinct_clarification_from_annotated_dimension(self):
        plan = {
            "status": "clarify",
            "measure_field": "COUNT_DISTINCT",
            "aggregation": "COUNT",
            "group_by": ["REGION"],
            "message": "Please confirm count of distinct regions.",
        }

        self.assertTrue(
            _resolve_distinct_dimension_count(
                plan,
                "How many regions are there in MBS system",
                sample_metadata(),
            )
        )
        self.assertEqual(plan["status"], "ready")
        self.assertEqual(plan["measure_field"], "REGION")
        self.assertEqual(plan["aggregation"], "COUNT_DISTINCT")
        self.assertEqual(plan["group_by"], [])

    def test_resolves_model_confirmation_for_valid_distinct_count_plan(self):
        plan = {
            "status": "clarify",
            "measure_field": "REGION",
            "aggregation": "COUNT_DISTINCT",
            "group_by": [],
            "message": "Please confirm the distinct region count.",
        }

        self.assertTrue(
            _resolve_distinct_dimension_count(
                plan,
                "How many regions are there in MBS system",
                sample_metadata(),
            )
        )
        self.assertEqual(plan["status"], "ready")
        self.assertEqual(plan["measure_field"], "REGION")
        self.assertEqual(plan["aggregation"], "COUNT_DISTINCT")

    def test_resolves_policy_count_from_annotated_policy_label(self):
        metadata = sample_metadata()
        metadata.source["PACKAGE BODY"] = metadata.source["PACKAGE BODY"].replace(
            "SELECT REGION, RECEIVED_AMOUNT",
            "SELECT REGION, RECEIVED_AMOUNT, DOCUMENT_CODE",
        )
        metadata.annotations.append(
            {
                "procedure": "MBS_DOCUMENT_DETAIL",
                "line": "4",
                "kind": "label",
                "value": "Policy Number | Policy | Document No | Document Code",
                "field": "DOCUMENT_CODE",
            }
        )
        plan = {
            "status": "clarify",
            "measure_field": None,
            "aggregation": None,
            "group_by": [],
            "message": "Please specify the measure.",
        }

        self.assertTrue(
            _resolve_distinct_dimension_count(
                plan,
                "How many policies are there",
                metadata,
            )
        )
        self.assertEqual(plan["status"], "ready")
        self.assertEqual(plan["measure_field"], "DOCUMENT_CODE")
        self.assertEqual(plan["aggregation"], "COUNT_DISTINCT")

    def test_policy_count_uses_live_metadata_without_calling_ollama(self):
        metadata = sample_metadata()
        metadata.source["PACKAGE BODY"] = metadata.source["PACKAGE BODY"].replace(
            "SELECT REGION, RECEIVED_AMOUNT",
            "SELECT REGION, RECEIVED_AMOUNT, DOCUMENT_CODE",
        )
        metadata.annotations.append(
            {
                "procedure": "MBS_DOCUMENT_DETAIL",
                "line": "4",
                "kind": "label",
                "value": "Policy Number | Policy | Document No | Document Code",
                "field": "DOCUMENT_CODE",
            }
        )
        with patch(
            "prompt_engine.urlopen",
            side_effect=AssertionError("Unique annotated count requests skip Ollama."),
        ):
            result = interpret_prompt("How many policies are there", metadata)

        self.assertEqual(result["operation"], "distinct_values")
        self.assertEqual(result["measure_field"], "DOCUMENT_CODE")
        self.assertEqual(result["aggregation"], "COUNT_DISTINCT")
        self.assertEqual(result["result_heading"], "Total Number of Policies")
        self.assertEqual(result["result_values_heading"], "Policies")

    def test_cash_value_document_code_routes_without_calling_ollama(self):
        metadata = sample_metadata()
        metadata.annotations.append(
            {
                "procedure": "MBS_DOCUMENT_DETAIL1",
                "line": "3",
                "kind": "capability",
                "value": "policy_cash_value",
            }
        )

        with patch(
            "prompt_engine.urlopen",
            side_effect=AssertionError("Ollama must not be called for this lookup."),
        ):
            result = interpret_prompt(
                "Okay let me have the cash values of MBT092026011140",
                metadata,
            )

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["operation"], "cash_value_lookup")
        self.assertEqual(result["document_code"], "MBT092026011140")
        self.assertEqual(result["procedure"], "MBS_DOCUMENT_DETAIL1")

    def test_cash_value_request_requires_exactly_one_document_code(self):
        metadata = sample_metadata()
        metadata.annotations.append(
            {
                "procedure": "MBS_DOCUMENT_DETAIL1",
                "line": "3",
                "kind": "capability",
                "value": "policy_cash_value",
            }
        )

        result = _cash_value_lookup("Show cash value for two policies", metadata)

        self.assertEqual(result["status"], "clarify")
        self.assertIn("exactly one", result["message"])

    def test_named_month_resolves_the_complete_inclusive_period(self):
        self.assertEqual(
            _parse_request_date_range("Region-wise business for August 2026"),
            {"P_SDATE": "2026-08-01", "P_EDATE": "2026-08-31 23:59:59"},
        )
        self.assertIsNone(_parse_request_date_range("Region-wise business"))

    def test_plan_rejects_fields_not_in_package_annotations(self):
        response = (
            '{"status":"ready","measure_field":"RECEIVED_AMOUNT",'
            '"aggregation":"SUM","group_by":["HR_SALARY"],"message":""}'
        )

        with self.assertRaisesRegex(ValueError, "not annotated"):
            _read_plan(response, sample_metadata())

    def test_unsupported_response_that_requests_clarification_is_normalized(self):
        response = (
            '{"status":"unsupported","measure_field":"RECEIVED_AMOUNT",'
            '"aggregation":"SUM","group_by":["REGION"],'
            '"message":"Please clarify which capability you want to use."}'
        )

        plan = _read_plan(response, sample_metadata())

        self.assertEqual(plan["status"], "clarify")
        self.assertEqual(plan["measure_field"], "RECEIVED_AMOUNT")
        self.assertEqual(plan["aggregation"], "SUM")
        self.assertEqual(plan["group_by"], ["REGION"])

    def test_exact_unique_annotated_label_resolves_measure_clarification(self):
        metadata = sample_metadata()
        metadata.annotations.append(
            {
                "procedure": "MBS_DOCUMENT_DETAIL",
                "line": "3",
                "kind": "label",
                "value": "Business | Contribution | Topup | Received Amount",
                "field": "RECEIVED_AMOUNT",
            }
        )
        plan = {
            "status": "clarify",
            "measure_field": "RECEIVED_AMOUNT",
            "aggregation": "SUM",
            "group_by": ["REGION"],
            "user_prompt": "Total region-wise business for August 2026",
        }

        self.assertTrue(_resolve_annotated_measure(plan, metadata))
        self.assertEqual(plan["status"], "ready")

    def test_duplicate_alias_does_not_resolve_measure_clarification(self):
        metadata = sample_metadata()
        metadata.source["PACKAGE BODY"] = metadata.source["PACKAGE BODY"].replace(
            "SELECT REGION, RECEIVED_AMOUNT",
            "SELECT REGION, RECEIVED_AMOUNT, TOPUP",
        )
        metadata.annotations.extend(
            [
                {
                    "procedure": "MBS_DOCUMENT_DETAIL",
                    "line": "3",
                    "kind": "label",
                    "value": "Topup",
                    "field": "RECEIVED_AMOUNT",
                },
                {
                    "procedure": "MBS_DOCUMENT_DETAIL",
                    "line": "4",
                    "kind": "label",
                    "value": "Topup",
                    "field": "TOPUP",
                },
            ]
        )
        plan = {
            "status": "clarify",
            "measure_field": "RECEIVED_AMOUNT",
            "aggregation": "SUM",
            "group_by": ["REGION"],
            "user_prompt": "Total region-wise Topup",
        }

        self.assertFalse(_resolve_annotated_measure(plan, metadata))
        self.assertEqual(plan["status"], "clarify")

    def test_composes_sql_from_package_query_and_approved_fields(self):
        plan = {
            "status": "ready",
            "measure_field": "RECEIVED_AMOUNT",
            "aggregation": "SUM",
            "group_by": ["REGION"],
            "message": "",
            "user_prompt": "Region-wise contribution for August 2026",
        }

        result = _build_sql(plan, sample_metadata())

        self.assertIn("SELECT QUERY_DATA.REGION", result["sql"])
        self.assertIn("SUM(QUERY_DATA.RECEIVED_AMOUNT)", result["sql"])
        self.assertIn("FROM BUSINESS_010226", result["sql"])
        self.assertIn(":P_SDATE", result["sql"])
        self.assertEqual(result["parameters"]["P_SDATE"], "2026-08-01")
        self.assertEqual(result["parameters"]["P_EDATE"], "2026-08-31 23:59:59")
        self.assertEqual(
            result["display_columns"],
            {"REGION": "Region", "RECEIVED_AMOUNT": "Contribution"},
        )
        self.assertEqual(result["measure_heading"], "Contribution")

    def test_composes_requested_report_columns_from_annotated_fields(self):
        metadata = sample_metadata()
        metadata.source["PACKAGE BODY"] = metadata.source["PACKAGE BODY"].replace(
            "SELECT REGION, RECEIVED_AMOUNT",
            "SELECT REGION, RECEIVED_AMOUNT, DOCUMENT_CODE, SUBMISSION_DATE",
        )
        metadata.annotations.extend(
            [
                {
                    "procedure": "MBS_DOCUMENT_DETAIL",
                    "line": "3",
                    "kind": "label",
                    "value": "Policy Number | Policy | Document Code",
                    "field": "DOCUMENT_CODE",
                },
                {
                    "procedure": "MBS_DOCUMENT_DETAIL",
                    "line": "4",
                    "kind": "label",
                    "value": "Submission Date | Date of Submission",
                    "field": "SUBMISSION_DATE",
                },
            ]
        )
        model_response = json.dumps(
            {
                "status": "ready",
                "measure_field": "RECEIVED_AMOUNT",
                "aggregation": "SUM",
                "group_by": ["SUBMISSION_DATE"],
                "select_columns": [
                    {
                        "field": "SUBMISSION_DATE",
                        "aggregation": None,
                        "alias": "Year",
                        "date_grain": "YEAR",
                    },
                    {
                        "field": "DOCUMENT_CODE",
                        "aggregation": "COUNT",
                        "alias": "Cases",
                        "date_grain": None,
                    },
                    {
                        "field": "RECEIVED_AMOUNT",
                        "aggregation": "SUM",
                        "alias": "Top up for the month",
                        "date_grain": None,
                    },
                ],
                "message": "",
            }
        )

        plan = _read_plan(model_response, metadata)
        result = _build_sql(
            {
                **plan,
                "user_prompt": (
                    "Business for September 2026, columns Year, Cases, "
                    "Top up for the month"
                ),
            },
            metadata,
        )

        self.assertIn("TRUNC(QUERY_DATA.SUBMISSION_DATE, 'YYYY')", result["sql"])
        self.assertIn('COUNT(QUERY_DATA.DOCUMENT_CODE) AS "Cases"', result["sql"])
        self.assertIn(
            'SUM(QUERY_DATA.RECEIVED_AMOUNT) AS "Top up for the month"',
            result["sql"],
        )
        self.assertEqual(result["parameters"]["P_SDATE"], "2026-09-01")

    def test_seeded_semantics_resolve_paraphrased_measure_and_registration_year(self):
        metadata = seeded_metadata()
        prompts = [
            "Show September 2026 business by policy registration year and region.",
            "How did September 2026 top-up break down across the years policies were registered?",
        ]

        for prompt in prompts:
            plan = {
                "status": "ready",
                "measure_field": "REGION",
                "aggregation": "COUNT",
                "group_by": ["REGION"],
                "user_prompt": prompt,
            }

            self.assertTrue(_apply_seeded_semantics(plan, prompt, metadata))
            result = _build_sql(plan, metadata)

            self.assertEqual(plan["measure_field"], "RECEIVED_AMOUNT")
            self.assertEqual(plan["aggregation"], "SUM")
            self.assertIn("TRUNC(QUERY_DATA.SUBMISSION_DATE, 'YYYY')", result["sql"])
            self.assertIn("SUM(QUERY_DATA.RECEIVED_AMOUNT)", result["sql"])
            self.assertEqual(result["parameters"]["P_SDATE"], "2026-09-01")

    def test_prompt_paraphrases_share_the_seeded_query_plan(self):
        metadata = seeded_metadata()
        model_plan = json.dumps(
            {
                "status": "ready",
                "measure_field": "RECEIVED_AMOUNT",
                "aggregation": "AVG",
                "group_by": ["SUBMISSION_DATE"],
                "select_columns": [],
                "message": "",
            }
        )
        prompts = [
            "Show me September 2026 business by policy registration year.",
            "How did September 2026 top-up break down across the years policies were registered?",
        ]

        for prompt in prompts:
            response = BytesIO(
                json.dumps({"message": {"content": model_plan}}).encode("utf-8")
            )
            with patch("prompt_engine.urlopen", return_value=response):
                result = interpret_prompt(prompt, metadata)

            self.assertIn(
                "TRUNC(QUERY_DATA.SUBMISSION_DATE, 'YYYY')",
                result["sql"],
            )
            self.assertIn("SUM(QUERY_DATA.RECEIVED_AMOUNT)", result["sql"])
            self.assertNotIn("AVG(QUERY_DATA.RECEIVED_AMOUNT)", result["sql"])
            self.assertEqual(result["parameters"]["P_SDATE"], "2026-09-01")

    def test_fully_seeded_output_columns_skip_ollama_for_paraphrases(self):
        metadata = seeded_metadata()
        prompts = [
            (
                "Show September 2026 business by policy registration year. "
                "Columns should be Year, Cases, Top up for the month.",
                "Top up for the month",
            ),
            (
                "Give September 2026 contribution across the years policies "
                "were registered. Output columns should be Year, Cases, "
                "Top-up for the month.",
                "Top-up for the month",
            ),
        ]

        for prompt, topup_alias in prompts:
            with patch(
                "prompt_engine.urlopen",
                side_effect=AssertionError(
                    "Unambiguous seeded output columns should not need Ollama."
                ),
            ):
                result = interpret_prompt(prompt, metadata)

            self.assertEqual(result["status"], "ready")
            self.assertEqual(
                result["select_columns"],
                [
                    {
                        "field": "SUBMISSION_DATE",
                        "aggregation": None,
                        "alias": "Year",
                        "date_grain": "YEAR",
                    },
                    {
                        "field": "DOCUMENT_CODE",
                        "aggregation": "COUNT",
                        "alias": "Cases",
                        "date_grain": None,
                    },
                    {
                        "field": "RECEIVED_AMOUNT",
                        "aggregation": "SUM",
                        "alias": topup_alias,
                        "date_grain": None,
                    },
                ],
            )
            self.assertIn("TRUNC(QUERY_DATA.SUBMISSION_DATE, 'YYYY')", result["sql"])
            self.assertIn("COUNT(QUERY_DATA.DOCUMENT_CODE)", result["sql"])
            self.assertIn("SUM(QUERY_DATA.RECEIVED_AMOUNT)", result["sql"])
            self.assertEqual(result["parameters"]["P_SDATE"], "2026-09-01")

    def test_optional_year_explanation_does_not_change_fast_plan(self):
        metadata = seeded_metadata()
        prompts = [
            (
                "Let me have the business of September 2026 and columns should be "
                "Year, Cases, Top up for the month, Year means policy register "
                "in which year?",
                ["Year", "Cases", "Top up for the month"],
            ),
            (
                "Let me have the business of September 2026 and columns should be "
                "Year, Cases, Top up for the month?",
                ["Year", "Cases", "Top up for the month"],
            ),
        ]
        plans = []

        for prompt, expected_aliases in prompts:
            self.assertEqual(_requested_output_aliases(prompt), expected_aliases)
            with patch(
                "prompt_engine.urlopen",
                side_effect=AssertionError(
                    "Seeded report columns must not wait for Ollama."
                ),
            ):
                result = interpret_prompt(prompt, metadata)

            self.assertEqual(result["status"], "ready")
            self.assertEqual(
                [
                    (
                        column["field"],
                        column["aggregation"],
                        column["alias"],
                        column["date_grain"],
                    )
                    for column in result["select_columns"]
                ],
                [
                    ("SUBMISSION_DATE", None, "Year", "YEAR"),
                    ("DOCUMENT_CODE", "COUNT", "Cases", None),
                    (
                        "RECEIVED_AMOUNT",
                        "SUM",
                        "Top up for the month",
                        None,
                    ),
                ],
            )
            plans.append(
                (
                    result["sql"],
                    result["parameters"],
                    result["select_columns"],
                )
            )

        self.assertEqual(plans[0], plans[1])

    def test_seeded_measure_and_group_phrasings_skip_ollama(self):
        metadata = seeded_metadata()
        prompts = [
            (
                "Show me region-wise business for August 2026.",
                ["REGION"],
                {},
            ),
            (
                "Break down August 2026 contribution across regions.",
                ["REGION"],
                {},
            ),
            (
                "How did September 2026 top-up break down across years policies were registered?",
                ["SUBMISSION_DATE"],
                {"SUBMISSION_DATE": "YEAR"},
            ),
        ]

        for prompt, groups, grains in prompts:
            with patch(
                "prompt_engine.urlopen",
                side_effect=AssertionError(
                    "Unambiguous seeded measures and groupings should not need Ollama."
                ),
            ):
                result = interpret_prompt(prompt, metadata)

            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["group_by"], groups)
            self.assertEqual(result["group_by_grains"], grains)
            self.assertIn("SUM(QUERY_DATA.RECEIVED_AMOUNT)", result["sql"])
            if grains:
                self.assertIn(
                    "TRUNC(QUERY_DATA.SUBMISSION_DATE, 'YYYY')",
                    result["sql"],
                )
            else:
                self.assertIn("QUERY_DATA.REGION", result["sql"])
            self.assertEqual(
                result["parameters"]["P_SDATE"],
                "2026-08-01" if "August" in prompt else "2026-09-01",
            )

    def test_selected_distinct_count_uses_oracle_count_distinct_syntax(self):
        metadata = seeded_metadata()
        plan = {
            "status": "ready",
            "measure_field": "DOCUMENT_CODE",
            "aggregation": "COUNT_DISTINCT",
            "group_by": ["SUBMISSION_DATE"],
            "select_columns": [
                {
                    "field": "SUBMISSION_DATE",
                    "aggregation": None,
                    "alias": "Year",
                    "date_grain": "YEAR",
                },
                {
                    "field": "DOCUMENT_CODE",
                    "aggregation": "COUNT_DISTINCT",
                    "alias": "Policies",
                    "date_grain": None,
                },
            ],
            "user_prompt": "September 2026 policy counts by registration year",
        }

        result = _build_sql(plan, metadata)

        self.assertIn(
            'COUNT(DISTINCT QUERY_DATA.DOCUMENT_CODE) AS "Policies"',
            result["sql"],
        )
        self.assertNotIn("COUNT_DISTINCT(", result["sql"])
        self.assertIsNone(result["value_sql"])
        self.assertIsNone(result["operation"])

    def test_grouped_distinct_count_uses_count_distinct_and_preserves_group(self):
        metadata = seeded_metadata()
        plan = {
            "status": "ready",
            "measure_field": "DOCUMENT_CODE",
            "aggregation": "COUNT_DISTINCT",
            "group_by": ["REGION"],
            "select_columns": [],
            "user_prompt": "Unique policies by region",
        }

        result = _build_sql(plan, metadata)

        self.assertIn(
            "COUNT(DISTINCT QUERY_DATA.DOCUMENT_CODE) AS DOCUMENT_CODE",
            result["sql"],
        )
        self.assertIn("GROUP BY QUERY_DATA.REGION", result["sql"])
        self.assertNotEqual(result["operation"], "distinct_values")
        self.assertIsNone(result["value_sql"])
        self.assertEqual(result["parameters"]["P_SDATE"], "0001-01-01")
        self.assertEqual(result["parameters"]["P_EDATE"], "9999-12-31 23:59:59")

    def test_resolves_model_column_choices_against_live_field_semantics(self):
        metadata = seeded_metadata()
        metadata.annotations.append(
            {
                "procedure": "MBS_DOCUMENT_DETAIL",
                "line": "5",
                "kind": "label",
                "value": "Topup | Policy Topup",
                "field": "TOPUP",
            }
        )
        prompt = (
            "September 2026 business insight. Columns should be Year, Cases, "
            "Top up for the month. Year means policy registration year."
        )
        plan = {
            "status": "clarify",
            "measure_field": "RECEIVED_AMOUNT",
            "aggregation": "SUM",
            "group_by": [],
            "user_prompt": prompt,
            "select_columns": [
                {
                    "field": "YEAR",
                    "aggregation": "COUNT",
                    "alias": "Year",
                    "date_grain": None,
                },
                {
                    "field": "RECEIVED_AMOUNT",
                    "aggregation": "COUNT",
                    "alias": "Cases",
                    "date_grain": "MONTH",
                },
                {
                    "field": "TOPUP",
                    "aggregation": "SUM",
                    "alias": "Top up",
                    "date_grain": "MONTH",
                },
            ],
        }

        self.assertTrue(_resolve_output_columns(plan, prompt, metadata))
        result = _build_sql(plan, metadata)

        self.assertEqual(
            plan["select_columns"],
            [
                {
                    "field": "SUBMISSION_DATE",
                    "aggregation": None,
                    "alias": "Year",
                    "date_grain": "YEAR",
                },
                {
                    "field": "DOCUMENT_CODE",
                    "aggregation": "COUNT",
                    "alias": "Cases",
                    "date_grain": None,
                },
                {
                    "field": "RECEIVED_AMOUNT",
                    "aggregation": "SUM",
                    "alias": "Top up for the month",
                    "date_grain": None,
                },
            ],
        )
        self.assertIn('COUNT(QUERY_DATA.DOCUMENT_CODE) AS "Cases"', result["sql"])
        self.assertIn(
            'SUM(QUERY_DATA.RECEIVED_AMOUNT) AS "Top up for the month"',
            result["sql"],
        )
        self.assertEqual(result["parameters"]["P_SDATE"], "2026-09-01")

    def test_composes_unbounded_count_distinct_as_a_dynamic_value_list(self):
        plan = {
            "status": "ready",
            "measure_field": "REGION",
            "aggregation": "COUNT_DISTINCT",
            "group_by": [],
            "message": "",
            "user_prompt": "How many regions are there in MBS system",
        }

        result = _build_sql(plan, sample_metadata())

        self.assertEqual(result["operation"], "distinct_values")
        self.assertIn("COUNT(DISTINCT QUERY_DATA.REGION)", result["sql"])
        self.assertIn("SELECT DISTINCT QUERY_DATA.REGION", result["value_sql"])
        self.assertEqual(result["parameters"]["P_SDATE"], "0001-01-01")
        self.assertEqual(result["parameters"]["P_EDATE"], "9999-12-31 23:59:59")
        self.assertEqual(result["required_parameters"], ["P_COMPANY_CODE"])
        self.assertEqual(result["result_heading"], "Total Number of Regions")
        self.assertEqual(result["result_values_heading"], "Regions")


if __name__ == "__main__":
    unittest.main()
