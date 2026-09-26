import oracledb

# Enable thick mode
oracledb.init_oracle_client(lib_dir=r"C:\instantclient_23_4")

def get_db_connection():
    connection = oracledb.connect(
        user="PQGTL",
        password="PQGTLADMIN$$",
        dsn="odah8-scan:1521/PQGTLODA8.PAKQATAR.COM.PK"
    )
    return connection
print(Connection)



