import re
import oracledb
import pandas as pd
from databases import get_db_connection

def fetch_live_catalog(database_name: str) -> pd.DataFrame:
    """Queries Oracle's system catalog directly using your connection profile."""
    catalog_query = r"""
        SELECT table_name, column_name, data_type 
        FROM user_tab_columns 
        WHERE table_name NOT LIKE 'BIN\$%'
    """
    conn = None
    try:
        conn = get_db_connection(database_name)
        return pd.read_sql(catalog_query, con=conn)
    finally:
        if conn:
            conn.close()

def generate_genius_sql(user_prompt: str, database_name: str = "PQTEPF", error_feedback: str = None) -> str:
    """
    DYNAMIC SMART-JOIN ENGINE: Builds SQL from thin air.
    Includes a Regex Input Validation layer to safely parse and auto-correct year digits.
    """
    catalog_df = fetch_live_catalog(database_name)
    
    if catalog_df.empty:
        raise ValueError(f"Database target '{database_name}' returned zero live structural tables.")
        
    prompt_lower = user_prompt.lower()

    # ===================================================
    # 🧠 DYNAMIC PARAMETER CONSTRAINT LAYER (YEAR CORRECTOR)
    # ===================================================
    # Look for any sequence of numbers inside the user's prompt string
    numbers_found = re.findall(r'\d+', prompt_lower)
    valid_year = None
    
    for num in numbers_found:
        if len(num) == 4:  # Perfect enterprise standard match (e.g., 2025)
            valid_year = num
        elif len(num) > 4: # Typos caught natively (e.g., 20255)
            # Auto-correct by trimming it down to the intended 4-digit corporate year block
            valid_year = num[:4]
            # Globally replace the typo inside the prompt with the clean year
            prompt_lower = prompt_lower.replace(num, valid_year)
    
    # ===================================================
    
    select_fields = []
    tables_to_use = set()
    group_fields = []
    where_clauses = []
    
    system_blacklist = ["qblock_name", "type", "id", "status"]
    
    for _, row in catalog_df.iterrows():
        tbl = row["TABLE_NAME"]
        col = row["COLUMN_NAME"]
        dtype = row["DATA_TYPE"]
        
        col_clean = col.lower().replace("_", " ")
        
        if col.lower() in system_blacklist:
            continue
            
        if any(word in prompt_lower for word in col_clean.split()) or any(word in col_clean for word in prompt_lower.split()):
            tables_to_use.add(tbl)
            alias = "p" if "MASTER" in tbl or "M_" in tbl or "POL" in tbl else "t"
            
            if dtype == "NUMBER":
                if error_feedback and "rollup" in error_feedback.lower():
                    select_fields.append(f"SUM({alias}.{col}) AS TOTAL_{col}")
                else:
                    if "WITH" in col or "TXN" in col or "LEAK" in col:
                        select_fields.append(f"SUM(CASE WHEN {alias}.TRANSACTION_TYPE = 'WITHDRAWAL' THEN {alias}.{col} ELSE 0 END) AS TOTAL_WITHDRAWALS")
                    elif "CONT" in col or "PREM" in col or "GROSS" in col:
                        select_fields.append(f"SUM({alias}.{col}) AS TOTAL_CONTRIBUTIONS")
                    else:
                        select_fields.append(f"SUM({alias}.{col}) AS TOTAL_{col}")
            elif "CODE" in col or "NAME" in col or "SEGMENT" in col or "YEAR" in col:
                select_fields.append(f"{alias}.{col} AS {col}")
                group_fields.append(f"{alias}.{col}")
                
                # Dynamic Filter Injection: If this column is a year column and we extracted a valid year parameter
                if "YEAR" in col and valid_year:
                    where_clauses.append(f"{alias}.{col} = '{valid_year}'")

    used_tables = list(tables_to_use)
    if not used_tables:
        raise ValueError("Could not dynamically resolve any matching database tables for this prompt.")
        
    if len(used_tables) == 1:
        from_clause = f"FROM {used_tables[0]} p"
    else:
        t1_cols = list(catalog_df[catalog_df["TABLE_NAME"] == used_tables[0]]["COLUMN_NAME"])
        t2_cols = list(catalog_df[catalog_df["TABLE_NAME"] == used_tables[1]]["COLUMN_NAME"])
        
        exact_intersect = list(set(t1_cols).intersection(set(t2_cols)))
        
        if exact_intersect:
            p_key = exact_intersect[0]
            t_key = exact_intersect[0]
        else:
            p_key = next((c for c in t1_cols if "POL" in c or "NO" in c), t1_cols[0])
            t_key = next((c for c in t2_cols if "POL" in c or "NO" in c or "ID" in c), t2_cols[0])
            
        from_clause = f"FROM {used_tables[0]} p \n\tLEFT JOIN {used_tables[1]} t ON p.{p_key} = t.{t_key}"

    select_str = ",\n\t".join(list(dict.fromkeys(select_fields)))
    group_str = f"GROUP BY {', '.join(list(dict.fromkeys(group_fields)))}" if group_fields else ""
    where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    order_str = f"ORDER BY 1 ASC" if group_fields else ""
    
    compiled_sql = f"""
SELECT 
\t{select_str}
{from_clause}
{where_str}
{group_str}
{order_str}
    """
    return compiled_sql.strip()
