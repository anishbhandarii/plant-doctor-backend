# create_admin.py — interactive script to create the first admin user
# Run on the server: python3 create_admin.py

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from auth import hash_password
from database import create_user

email     = input("Admin email: ").strip()
password  = input("Admin password: ").strip()
full_name = input("Full name: ").strip()

if len(password) < 8:
    print("Error: Password must be at least 8 characters")
    sys.exit(1)

try:
    user = create_user(
        email=email,
        password_hash=hash_password(password),
        full_name=full_name,
        role="admin",
        preferred_language="english",
    )
    print(f"Admin created: {user['email']} (id={user['id']})")
except ValueError as e:
    print(f"Error: {e}")
    sys.exit(1)
