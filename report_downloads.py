"""Convert query result dataframes to local report download formats."""

from io import BytesIO

import pandas as pd


def dataframe_to_excel(dataframe: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="Business Insights")
    return output.getvalue()


def dataframe_to_csv(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8")
