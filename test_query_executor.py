"""Tests for safe execution and typed binding of generated queries."""

import unittest
from datetime import datetime
from unittest.mock import patch

from cancellation import CancellationToken, PromptCancelled
from query_executor import _bind_values, execute_generated_query


class QueryExecutorTests(unittest.TestCase):
    def test_cancellation_interrupts_running_query_and_closes_connection(self):
        class Cursor:
            description = [("TOTAL",)]

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def execute(self, _sql, _binds=None):
                cancellation.cancel()

        class Connection:
            cancelled = False
            closed = False

            def cursor(self):
                return Cursor()

            def cancel(self):
                self.cancelled = True

            def close(self):
                self.closed = True

        cancellation = CancellationToken()
        connection = Connection()
        result = {
            "status": "ready",
            "sql": "SELECT COUNT(*) AS TOTAL FROM EPF_UW_DOCUMENT",
            "parameters": {},
            "display_columns": {},
        }
        with patch("query_executor.get_db_connection", return_value=connection):
            with self.assertRaises(PromptCancelled):
                execute_generated_query(result, cancellation=cancellation)

        self.assertTrue(connection.cancelled)
        self.assertTrue(connection.closed)

    def test_binds_dates_as_oracle_datetime_and_company_code_one(self):
        result = {
            "parameters": {
                "P_SDATE": "2026-08-01",
                "P_EDATE": "2026-08-31 23:59:59",
            }
        }
        binds = _bind_values(result)

        self.assertEqual(binds["P_SDATE"], datetime(2026, 8, 1))
        self.assertEqual(binds["P_EDATE"], datetime(2026, 8, 31, 23, 59, 59))
        self.assertEqual(binds["P_COMPANY_CODE"], 1)

    def test_company_code_defaults_to_user_approved_qa_value(self):
        binds = _bind_values({"parameters": {}})
        self.assertEqual(binds["P_COMPANY_CODE"], 1)

    def test_binds_seeded_cnic_lookup_value(self):
        binds = _bind_values({"parameters": {"P_CNIC": "4220165562223"}})

        self.assertEqual(binds["P_CNIC"], "4220165562223")
        self.assertEqual(binds["P_COMPANY_CODE"], 1)

    def test_executes_seeded_direct_policy_count(self):
        class Cursor:
            description = [("Total Number of Policies",)]

            def __init__(self):
                self.statements = []

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def execute(self, sql, binds=None):
                self.statements.append((sql, binds))

            def fetchall(self):
                return [(4000,)]

        class Connection:
            def __init__(self):
                self.test_cursor = Cursor()

            def cursor(self):
                return self.test_cursor

            def close(self):
                pass

        connection = Connection()
        result = {
            "status": "ready",
            "operation": "direct_count",
            "sql": (
                'SELECT COUNT(DOCUMENT_ID_PK) AS "Total Number of Policies" '
                "FROM EPF_UW_DOCUMENT"
            ),
            "parameters": {},
            "display_columns": {},
        }
        with patch("query_executor.get_db_connection", return_value=connection):
            dataframe = execute_generated_query(result)

        self.assertEqual(
            connection.test_cursor.statements[1][0],
            result["sql"],
        )
        self.assertEqual(dataframe.iloc[0]["Total Number of Policies"], 4000)

    def test_executes_seeded_policy_lookup_with_bound_cnic(self):
        class Cursor:
            description = [("Policy Number",)]

            def __init__(self):
                self.statements = []

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def execute(self, sql, binds=None):
                self.statements.append((sql, binds))

            def fetchall(self):
                return [("MBT092026011140",)]

        class Connection:
            def __init__(self):
                self.test_cursor = Cursor()

            def cursor(self):
                return self.test_cursor

            def close(self):
                pass

        connection = Connection()
        result = {
            "status": "ready",
            "operation": "lookup",
            "sql": (
                'SELECT DISTINCT T1.DOCUMENT_CODE AS "Policy Number" '
                "FROM EPF_UW_DOCUMENT T1 "
                "JOIN EPF_UW_EMPLOYEES T2 "
                "ON T1.DOCUMENT_ID_PK = T2.DOCUMENT_ID_FK "
                "WHERE T2.CNIC_NO = :P_CNIC "
                "AND T1.COMPANYID_FK = :P_COMPANY_CODE"
            ),
            "parameters": {"P_CNIC": "4220165562223"},
            "display_columns": {},
        }
        with patch("query_executor.get_db_connection", return_value=connection):
            dataframe = execute_generated_query(result)

        self.assertEqual(
            connection.test_cursor.statements[1][1]["P_CNIC"],
            "4220165562223",
        )
        self.assertEqual(
            connection.test_cursor.statements[1][1]["P_COMPANY_CODE"],
            1,
        )
        self.assertEqual(dataframe["Policy Number"].tolist(), ["MBT092026011140"])

    def test_rejects_non_select_before_opening_connection(self):
        result = {"status": "ready", "sql": "DELETE FROM BUSINESS_010226"}
        with patch("query_executor.get_db_connection") as connect:
            with self.assertRaisesRegex(ValueError, "read-only SELECT"):
                execute_generated_query(result)
        connect.assert_not_called()

    def test_runs_generated_query_in_read_only_transaction(self):
        class Cursor:
            description = [("REGION",), ("RECEIVED_AMOUNT",)]

            def __init__(self):
                self.statements = []

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def execute(self, sql, binds=None):
                self.statements.append((sql, binds))

            def fetchall(self):
                return [("Lahore Region", 100)]

        class Connection:
            def __init__(self):
                self.test_cursor = Cursor()
                self.closed = False

            def cursor(self):
                return self.test_cursor

            def close(self):
                self.closed = True

        connection = Connection()
        result = {
            "status": "ready",
            "sql": "SELECT REGION, SUM(RECEIVED_AMOUNT) FROM MBS_DATA "
            "WHERE TRANS_DATE BETWEEN :P_SDATE AND :P_EDATE "
            "AND COMPANYID_FK = :P_COMPANY_CODE GROUP BY REGION",
            "parameters": {
                "P_SDATE": "2026-08-01",
                "P_EDATE": "2026-08-31 23:59:59",
            },
            "display_columns": {"REGION": "Region", "RECEIVED_AMOUNT": "Business"},
        }
        with patch("query_executor.get_db_connection", return_value=connection):
            dataframe = execute_generated_query(result)

        self.assertEqual(connection.test_cursor.statements[0][0], "SET TRANSACTION READ ONLY")
        self.assertEqual(len(dataframe), 1)
        self.assertEqual(dataframe.iloc[0]["Region"], "Lahore Region")
        self.assertNotIn("RECEIVED_AMOUNT", dataframe.columns)
        self.assertEqual(dataframe.index.tolist(), [1])
        self.assertTrue(connection.closed)

    def test_distinct_count_fetches_values_only_for_small_result_sets(self):
        class Cursor:
            description = [("DISTINCT_COUNT",)]

            def __init__(self):
                self.executed = []

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def execute(self, sql, binds=None):
                self.executed.append(sql)

            def fetchone(self):
                return (2,)

            def fetchall(self):
                self.description = [("REGION",)]
                return [("North",), ("South",)]

        class Connection:
            def __init__(self):
                self.test_cursor = Cursor()

            def cursor(self):
                return self.test_cursor

            def close(self):
                pass

        connection = Connection()
        result = {
            "status": "ready",
            "operation": "distinct_values",
            "sql": "SELECT COUNT(DISTINCT REGION) AS DISTINCT_COUNT FROM MBS_DATA",
            "value_sql": "SELECT DISTINCT REGION FROM MBS_DATA",
            "result_heading": "Total Number of Regions",
            "result_values_heading": "Regions",
            "parameters": {},
        }
        with patch("query_executor.get_db_connection", return_value=connection):
            dataframe = execute_generated_query(result)

        self.assertEqual(result["distinct_count"], 2)
        self.assertEqual(dataframe["Regions"].tolist(), ["North", "South"])
        self.assertEqual(dataframe.index.tolist(), [1, 2])
        self.assertEqual(len(connection.test_cursor.executed), 3)

    def test_distinct_count_does_not_fetch_large_value_lists(self):
        class Cursor:
            description = [("DISTINCT_COUNT",)]

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def execute(self, _sql, _binds=None):
                pass

            def fetchone(self):
                return (1000,)

            def fetchall(self):
                raise AssertionError("Large distinct values must not be fetched.")

        class Connection:
            def cursor(self):
                return Cursor()

            def close(self):
                pass

        result = {
            "status": "ready",
            "operation": "distinct_values",
            "sql": "SELECT COUNT(DISTINCT CODE) AS DISTINCT_COUNT FROM MBS_DATA",
            "value_sql": "SELECT DISTINCT CODE FROM MBS_DATA",
            "result_heading": "Total Number of Policies",
            "result_values_heading": "Policies",
            "parameters": {},
        }
        with patch("query_executor.get_db_connection", return_value=Connection()):
            dataframe = execute_generated_query(result)

        self.assertEqual(result["distinct_count"], 1000)
        self.assertEqual(dataframe.iloc[0]["Total Number of Policies"], 1000)
        self.assertEqual(dataframe.index.tolist(), [1])

    def test_cash_value_lookup_resolves_document_and_calls_annotated_procedure(self):
        class Variable:
            def __init__(self):
                self.value = None

            def getvalue(self):
                return self.value

            def setvalue(self, _position, value):
                self.value = value

        class Cursor:
            def __init__(self):
                self.statements = []

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def execute(self, sql, binds=None):
                self.statements.append((sql, binds))

            def fetchall(self):
                return [(37144, 153603, "Faisal", datetime(2026, 10, 13))]

            def var(self, _type):
                return Variable()

            def callproc(self, name, parameters):
                self.procedure_name = name
                self.parameters = parameters
                parameters[7].setvalue(0, 1098.14)

        class Connection:
            def __init__(self):
                self.test_cursor = Cursor()
                self.closed = False

            def cursor(self):
                return self.test_cursor

            def close(self):
                self.closed = True

        connection = Connection()
        result = {
            "status": "ready",
            "operation": "cash_value_lookup",
            "owner": "PQTEPF",
            "package": "PKG_BIEL_MBS",
            "procedure": "MBS_DOCUMENT_DETAIL1",
            "document_code": "MBT092026011140",
        }
        with patch("query_executor.get_db_connection", return_value=connection):
            dataframe = execute_generated_query(result)

        self.assertEqual(connection.test_cursor.procedure_name, "PQTEPF.PKG_BIEL_MBS.MBS_DOCUMENT_DETAIL1")
        self.assertEqual(connection.test_cursor.parameters[5], None)
        self.assertEqual(connection.test_cursor.parameters[6], "CASH_VALUE")
        self.assertEqual(dataframe.iloc[0]["Policy Holder"], "Faisal")
        self.assertEqual(dataframe.iloc[0]["Cash Value"], 1098.14)
        self.assertTrue(connection.closed)


if __name__ == "__main__":
    unittest.main()
