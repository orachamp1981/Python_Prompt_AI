"""Execute generated, read-only package queries with typed Oracle binds."""

import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import oracledb
import pandas as pd

from cancellation import CancellationToken
from databases import get_db_connection


DEFAULT_QA_COMPANY_CODE = 1
BIND_PATTERN = re.compile(r":([A-Z][A-Z0-9_$#]*)", re.IGNORECASE)
SUPPORTED_BINDS = {"P_SDATE", "P_EDATE", "P_COMPANY_CODE"}
IDENTIFIER_PATTERN = re.compile(r"^[A-Z][A-Z0-9_$#]*$")
MAX_DISTINCT_VALUES_TO_DISPLAY = 100


def _bind_values(result: Dict[str, Any]) -> Dict[str, Any]:
    parameters = result.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("Generated query parameters must be an object.")

    binds = {}
    for name in ("P_SDATE", "P_EDATE"):
        value = parameters.get(name)
        if value is not None:
            if not isinstance(value, str):
                raise ValueError(f"Generated {name} value must be an ISO date.")
            try:
                binds[name] = datetime.fromisoformat(value)
            except ValueError as error:
                raise ValueError(f"Generated {name} value is not a valid ISO date.") from error

    for name, value in parameters.items():
        normalized_name = str(name).upper()
        if normalized_name in {"P_SDATE", "P_EDATE"}:
            continue
        if not IDENTIFIER_PATTERN.fullmatch(normalized_name):
            raise ValueError("Generated query has an invalid bind name.")
        if not isinstance(value, (str, int, float)):
            raise ValueError(f"Generated {normalized_name} value has an invalid type.")
        binds[normalized_name] = value

    binds.setdefault("P_COMPANY_CODE", DEFAULT_QA_COMPANY_CODE)

    return binds


def execute_generated_query(
    result: Dict[str, Any],
    database_name: str = "PQTEPF",
    cancellation: Optional[CancellationToken] = None,
) -> pd.DataFrame:
    """Execute only a ready generated SELECT inside a read-only transaction."""
    if result.get("status") != "ready":
        raise ValueError("Only a ready query operation can be executed.")
    if result.get("operation") == "cash_value_lookup":
        return _execute_cash_value_lookup(result, database_name, cancellation)

    sql = result.get("sql")
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("Only a ready generated query can be executed.")
    if not re.match(r"^\s*SELECT\b", sql, re.IGNORECASE) or ";" in sql:
        raise ValueError("Only one read-only SELECT statement can be executed.")

    uncommented_sql = re.sub(r"--[^\n]*|/\*[\s\S]*?\*/", "", sql)
    bind_names = {name.upper() for name in BIND_PATTERN.findall(uncommented_sql)}
    supplied_parameters = result.get("parameters", {})
    if not isinstance(supplied_parameters, dict):
        raise ValueError("Generated query parameters must be an object.")
    supported_binds = SUPPORTED_BINDS | {
        str(name).upper() for name in supplied_parameters
    }
    unsupported_binds = bind_names - supported_binds
    if unsupported_binds:
        raise ValueError(
            "Generated query has unsupported bind parameters: "
            + ", ".join(sorted(unsupported_binds))
        )

    binds = _bind_values(result)
    missing_binds = bind_names - binds.keys()
    if missing_binds:
        raise ValueError(
            "Generated query is missing bind values: " + ", ".join(sorted(missing_binds))
        )

    connection = get_db_connection(database_name)
    if cancellation is not None:
        cancellation.attach_cancel_callback(connection.cancel)
    try:
        if cancellation is not None:
            cancellation.check()
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            if cancellation is not None:
                cancellation.check()
            cursor.execute(sql, binds)
            if cancellation is not None:
                cancellation.check()
            if result.get("operation") == "distinct_values":
                count_row = cursor.fetchone()
                distinct_count = int(count_row[0] or 0)
                result["distinct_count"] = distinct_count
                if distinct_count <= MAX_DISTINCT_VALUES_TO_DISPLAY:
                    value_sql = result.get("value_sql")
                    if not isinstance(value_sql, str) or not re.match(
                        r"^\s*SELECT\b", value_sql, re.IGNORECASE
                    ) or ";" in value_sql:
                        raise ValueError("Generated distinct-values query is invalid.")
                    cursor.execute(value_sql, binds)
                    rows = cursor.fetchall()
                    columns = [
                        result.get("result_values_heading")
                        or description[0]
                        for description in cursor.description
                    ]
                else:
                    rows = [(distinct_count,)]
                    columns = [
                        result.get("result_heading")
                        or "Total Number of Distinct Values"
                    ]
            else:
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
    finally:
        if cancellation is not None:
            cancellation.clear_cancel_callback()
        connection.close()

    dataframe = pd.DataFrame(rows, columns=columns)
    display_columns = result.get("display_columns", {})
    if not isinstance(display_columns, dict):
        raise ValueError("Generated result display labels must be an object.")
    dataframe = dataframe.rename(columns=display_columns)
    dataframe.index = range(1, len(dataframe) + 1)
    dataframe.index.name = "No."
    return dataframe


def _execute_cash_value_lookup(
    result: Dict[str, Any],
    database_name: str,
    cancellation: Optional[CancellationToken] = None,
) -> pd.DataFrame:
    document_code = result.get("document_code")
    if not isinstance(document_code, str) or not re.fullmatch(
        r"[A-Z]{2,}\d{6,}", document_code, re.IGNORECASE
    ):
        raise ValueError("Cash-value lookup requires a valid policy or document code.")

    owner = str(result.get("owner", "")).upper()
    package = str(result.get("package", "")).upper()
    procedure = str(result.get("procedure", "")).upper()
    if not all(
        IDENTIFIER_PATTERN.fullmatch(identifier)
        for identifier in (owner, package, procedure)
    ):
        raise ValueError("Cash-value lookup has invalid package metadata.")

    connection = get_db_connection(database_name)
    if cancellation is not None:
        cancellation.attach_cancel_callback(connection.cancel)
    try:
        if cancellation is not None:
            cancellation.check()
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute(
                """
                SELECT DISTINCT
                    T1.DOCUMENT_ID_PK,
                    T2.EMPLOYEE_ID_PK,
                    T3.PARTICIPANT_NAME,
                    TRUNC(SYSDATE) + 1 AS PROCEDURE_END_DATE
                FROM EPF_UW_DOCUMENT T1
                JOIN EPF_UW_EMPLOYEES T2
                    ON T1.DOCUMENT_ID_PK = T2.DOCUMENT_ID_FK
                LEFT JOIN EPF_GEN_ST_PARTICIPANT T3
                    ON T1.PARTICIPANT_ID_FK = T3.PARTICIPANT_ID_PK
                WHERE T1.DOCUMENT_CODE = :document_code
                  AND T1.COMPANYID_FK = :company_code
                """,
                {
                    "document_code": document_code,
                    "company_code": DEFAULT_QA_COMPANY_CODE,
                },
            )
            matches = cursor.fetchall()
            if cancellation is not None:
                cancellation.check()
            if not matches:
                raise LookupError(
                    f"No policy was found for document code {document_code}."
                )
            if len(matches) != 1:
                raise LookupError(
                    f"Document code {document_code} maps to multiple employee records."
                )

            document_id, employee_id, participant_name, procedure_end_date = matches[0]
            cash_value = cursor.var(oracledb.NUMBER)
            agent_name = cursor.var(str)
            cursor.callproc(
                f"{owner}.{package}.{procedure}",
                [
                    procedure_end_date - timedelta(days=1),
                    procedure_end_date,
                    DEFAULT_QA_COMPANY_CODE,
                    document_id,
                    employee_id,
                    None,
                    "CASH_VALUE",
                    cash_value,
                    agent_name,
                ],
            )
            value = cash_value.getvalue()
    finally:
        if cancellation is not None:
            cancellation.clear_cancel_callback()
        connection.close()

    return pd.DataFrame(
        [
            {
                "Document Code": document_code,
                "Policy Holder": participant_name,
                "Cash Value": value,
                "As Of": (procedure_end_date - timedelta(days=1)).date(),
            }
        ]
    )
