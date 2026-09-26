import io
import pandas as pd

def convert_dataframe_to_excel(df: pd.DataFrame) -> bytes:
    """Compiles in-memory data frames directly into clean binary Excel streams."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Business Insights')
    return output.getvalue()

def convert_dataframe_to_csv(df: pd.DataFrame) -> str:
    """Compiles structured tables to standard CSV text layout."""
    return df.to_csv(index=False).encode('utf-8')
