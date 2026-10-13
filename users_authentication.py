"""Temporary in-memory accounts for local role-based QA."""

import os
from typing import Dict, Optional


TEST_USERS = {
    "admin": {"role": "admin", "password_env": "PROMPT_AI_ADMIN_PASSWORD"},
    "developer": {"role": "developer", "password_env": "PROMPT_AI_DEVELOPER_PASSWORD"},
    "user": {"role": "normal_user", "password_env": "PROMPT_AI_USER_PASSWORD"},
}
DEFAULT_TEST_PASSWORD = "123"
GUEST_ROLE = "guest"


def authenticate(username: str, password: str) -> Optional[Dict[str, str]]:
    """Authenticate one temporary QA account; return its normalized identity."""
    normalized_username = username.strip().lower()
    account = TEST_USERS.get(normalized_username)
    if not account:
        return None

    expected_password = os.environ.get(account["password_env"], DEFAULT_TEST_PASSWORD)
    if password != expected_password:
        return None

    return {"username": normalized_username, "role": account["role"]}


def can_view_sql(role: str) -> bool:
    """Generated SQL is visible only to admins and developers."""
    return role in {"admin", "developer"}
