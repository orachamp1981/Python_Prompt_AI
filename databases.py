import oracledb
import pandas as pd  # FIXED: Added the missing Pandas import cleanly at the top

oracledb.init_oracle_client(
    lib_dir=r"C:\instantclient_23_4"
)

DATABASES = {
    "PQTEPF": {
        "user": "PQTEPF",
        "password": "PQTEPF",
        "dsn": "odah8-scan.pakqatar.com.pk:1521/pqtepf_app.pakqatar.com.pk"
    },
    "PQGTL": {
        "user": "PQGTL",
        "password": "PQGTLADMIN$$",
        "dsn": "odah8-scan:1521/PQGTLODA8.PAKQATAR.COM.PK"
    }
}

def get_db_connection(database_name: str):
    database_name = database_name.upper()

    if database_name not in DATABASES:
        raise ValueError(f"Unsupported database: {database_name}")

    config = DATABASES[database_name]

    return oracledb.connect(
        user=config["user"],
        password=config["password"],
        dsn=config["dsn"]
    )

def execute_pqtepf_query(sql_query: str) -> pd.DataFrame:
    """Helper function to cleanly fetch data frames using your connection rule."""
    import pandas as pd
    connection = None
    try:
        connection = get_db_connection("PQTEPF")
        df = pd.read_sql(sql_query, con=connection)
        return df
    finally:
        if connection:
            connection.close()
