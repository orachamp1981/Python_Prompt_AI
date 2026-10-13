"""Create connections to the configured Oracle databases."""

import os

import oracledb


# 1. Default Oracle routes
DEFAULT_DSNS = {
    "PQTEPF": "PQTEPFODA",
}


# 2. Optional Oracle client setup
client_library = os.environ.get("ORACLE_CLIENT_LIB_DIR")
if client_library:
    oracledb.init_oracle_client(lib_dir=client_library)


# 3. Database connection
def get_db_connection(database_name: str) -> oracledb.Connection:
    """Connect using <DATABASE>_USER, <DATABASE>_PASSWORD, and <DATABASE>_DSN."""
    name = database_name.strip().upper()
    setting_names = {
        "user": f"{name}_USER",
        "password": f"{name}_PASSWORD",
    }
    settings = {key: os.environ.get(env_name) for key, env_name in setting_names.items()}
    dsn_setting = f"{name}_DSN"
    dsn = os.environ.get(dsn_setting) or DEFAULT_DSNS.get(name)
    if dsn:
        settings["dsn"] = dsn

    missing = [setting_names[key] for key, value in settings.items() if not value]
    if "dsn" not in settings:
        missing.append(dsn_setting)

    if missing:
        raise ValueError(
            f"Missing Oracle connection settings: {', '.join(missing)}"
        )

    return oracledb.connect(**settings)
