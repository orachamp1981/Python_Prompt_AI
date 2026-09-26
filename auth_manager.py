# Simple internal mock directory for testing local organization deployment
USER_DIRECTORY = {
    "admin": {"password": "123", "role": "EXECUTIVE"},
    "user1": {"password": "123", "role": "UNDERWRITING"},
    "user2": {"password": "123", "role": "FINANCE"}
}

def verify_user_credentials(username, password):
    if username in USER_DIRECTORY and USER_DIRECTORY[username]["password"] == password:
        return {"authenticated": True, "role": USER_DIRECTORY[username]["role"]}
    return {"authenticated": False, "role": None}
