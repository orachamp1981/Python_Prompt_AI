import oracledb

# Enable thick mode
oracledb.init_oracle_client(lib_dir=r"C:\instantclient_23_4")

def get_db_connection():
    connection = oracledb.connect(
        user="PQTEPF",
        password="PQTEPF",
        dsn="odah8-scan.pakqatar.com.pk:1521/pqtepf_app.pakqatar.com.pk"
    )
    return connection