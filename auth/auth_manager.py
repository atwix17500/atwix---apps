import hashlib
import re
import secrets
from datetime import datetime, timedelta

from auth.database import get_connection, init_db

SESSION_TIMEOUT_MINUTES = 30

PASSWORD_RULES = {
    "min_length": 8,
    "require_uppercase": True,
    "require_lowercase": True,
    "require_digit": True,
    "require_special": True,
}

USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_]{3,20}$")
EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


class AuthManager:
    def __init__(self):
        conn = get_connection()
        init_db(conn)
        self._seed_default_admin(conn)
        conn.close()

    def _seed_default_admin(self, conn):
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if count > 0:
            return

        conn.execute(
            """
            INSERT INTO users (username, email, password_hash, role, country, district, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "admin",
                "admin@croprec.com",
                self._hash_password("Admin@123"),
                "admin",
                "Not set",
                "Not set",
                datetime.now().isoformat(),
            ),
        )
        conn.commit()

    def _hash_password(self, password: str, salt: str = None) -> str:
        if salt is None:
            salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000
        )
        return f"{salt}${hashed.hex()}"

    def _verify_password(self, password: str, stored_hash: str) -> bool:
        try:
            salt, _ = stored_hash.split("$", 1)
            return self._hash_password(password, salt) == stored_hash
        except ValueError:
            return False

    def validate_username(self, username: str) -> tuple[bool, str]:
        username = username.strip()
        if not username:
            return False, "Username is required."
        if not USERNAME_PATTERN.match(username):
            return False, "Username must be 3–20 characters (letters, numbers, underscore only)."
        return True, ""

    def validate_email(self, email: str) -> tuple[bool, str]:
        email = email.strip()
        if not email:
            return False, "Email is required."
        if not EMAIL_PATTERN.match(email):
            return False, "Please enter a valid email address."
        return True, ""

    def validate_location(self, country: str, district: str) -> tuple[bool, str]:
        country = country.strip()
        district = district.strip()
        if not country or country == "Select country":
            return False, "Please select your country."
        if not district:
            return False, "District / region is required."
        if len(district) < 2:
            return False, "District / region must be at least 2 characters."
        if len(district) > 80:
            return False, "District / region is too long (max 80 characters)."
        return True, ""

    def validate_password(self, password: str) -> tuple[bool, list[str]]:
        errors = []
        if len(password) < PASSWORD_RULES["min_length"]:
            errors.append(f"At least {PASSWORD_RULES['min_length']} characters")
        if PASSWORD_RULES["require_uppercase"] and not re.search(r"[A-Z]", password):
            errors.append("At least one uppercase letter")
        if PASSWORD_RULES["require_lowercase"] and not re.search(r"[a-z]", password):
            errors.append("At least one lowercase letter")
        if PASSWORD_RULES["require_digit"] and not re.search(r"\d", password):
            errors.append("At least one number")
        if PASSWORD_RULES["require_special"] and not re.search(
            r"[!@#$%^&*(),.?\":{}|<>]", password
        ):
            errors.append("At least one special character (!@#$%^&*...)")
        return len(errors) == 0, errors

    def register(
        self,
        username: str,
        email: str,
        password: str,
        confirm_password: str,
        country: str,
        district: str,
    ) -> tuple[bool, str]:
        valid, msg = self.validate_username(username)
        if not valid:
            return False, msg

        valid, msg = self.validate_email(email)
        if not valid:
            return False, msg

        valid, msg = self.validate_location(country, district)
        if not valid:
            return False, msg

        valid, errors = self.validate_password(password)
        if not valid:
            return False, "Password requirements not met: " + ", ".join(errors)

        if password != confirm_password:
            return False, "Passwords do not match."

        username = username.lower()
        email = email.lower()
        conn = get_connection()

        if conn.execute(
            "SELECT 1 FROM users WHERE username = ?", (username,)
        ).fetchone():
            conn.close()
            return False, "Username already exists."

        if conn.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone():
            conn.close()
            return False, "Email is already registered."

        conn.execute(
            """
            INSERT INTO users (username, email, password_hash, role, country, district, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                username,
                email,
                self._hash_password(password),
                "user",
                country.strip(),
                district.strip(),
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
        conn.close()
        return True, "Account created successfully. You can now log in."

    def login(self, username: str, password: str) -> tuple[bool, str, dict | None]:
        if not username.strip() or not password:
            return False, "Please enter both username and password.", None

        username = username.strip().lower()
        conn = get_connection()
        row = conn.execute(
            """
            SELECT id, username, email, password_hash, role, country, district
            FROM users WHERE username = ?
            """,
            (username,),
        ).fetchone()
        conn.close()

        if row is None or not self._verify_password(password, row["password_hash"]):
            return False, "Invalid username or password.", None

        return True, "Login successful.", {
            "user_id": row["id"],
            "username": row["username"],
            "email": row["email"],
            "role": row["role"],
            "country": row["country"] or "Not set",
            "district": row["district"] or "Not set",
            "login_time": datetime.now().isoformat(),
        }

    def get_user_profile(self, user_id: int) -> dict | None:
        conn = get_connection()
        row = conn.execute(
            """
            SELECT username, email, role, country, district, created_at
            FROM users WHERE id = ?
            """,
            (user_id,),
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def save_prediction(
        self,
        user_id: int,
        nitrogen: float,
        phosphorus: float,
        potassium: float,
        temperature: float,
        humidity: float,
        ph: float,
        rainfall: float,
        recommended_crop: str,
    ):
        conn = get_connection()
        conn.execute(
            """
            INSERT INTO predictions (
                user_id, nitrogen, phosphorus, potassium,
                temperature, humidity, ph, rainfall,
                recommended_crop, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                nitrogen,
                phosphorus,
                potassium,
                temperature,
                humidity,
                ph,
                rainfall,
                recommended_crop,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
        conn.close()

    def get_user_predictions(self, user_id: int, limit: int = 10) -> list[dict]:
        conn = get_connection()
        rows = conn.execute(
            """
            SELECT recommended_crop, nitrogen, phosphorus, potassium,
                   temperature, humidity, ph, rainfall, created_at
            FROM predictions
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def is_session_valid(self, session: dict | None) -> bool:
        if not session or "login_time" not in session:
            return False
        login_time = datetime.fromisoformat(session["login_time"])
        expiry = login_time + timedelta(minutes=SESSION_TIMEOUT_MINUTES)
        return datetime.now() < expiry

    def get_session_remaining(self, session: dict) -> int:
        login_time = datetime.fromisoformat(session["login_time"])
        expiry = login_time + timedelta(minutes=SESSION_TIMEOUT_MINUTES)
        remaining = (expiry - datetime.now()).total_seconds()
        return max(0, int(remaining // 60))
