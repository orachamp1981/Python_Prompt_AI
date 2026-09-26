import os
from dotenv import load_dotenv

load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT")
ORACLE_CLIENT_PATH = os.getenv("ORACLE_CLIENT_PATH")

print("Environment       :", ENVIRONMENT)
print("Oracle Client Path:", ORACLE_CLIENT_PATH)