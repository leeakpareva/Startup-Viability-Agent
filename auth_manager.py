"""
Authentication Manager Module for NAVADA
Handles user authentication, session management, and conversation storage.
"""

import hashlib
import secrets
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class AuthManager:
    """Simple authentication manager with SQLite backend."""

    def __init__(self, db_path: str = "navada_auth.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database tables."""
        try:
            # Use timeout to prevent hanging
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.cursor()

                # Set pragmas for better performance
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")

                # Users table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        username TEXT UNIQUE NOT NULL,
                        email TEXT UNIQUE,
                        password_hash TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_token TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(user_id)
                    )
                ''')

                # Conversations table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS conversations (
                        conversation_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        thread_id TEXT,
                        mode TEXT,
                        messages TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(user_id)
                    )
                ''')

                # User actions log table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_actions (
                        action_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        action_type TEXT,
                        details TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(user_id)
                    )
                ''')

                conn.commit()
                logger.info("Database tables initialized successfully")
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                logger.error("Database is locked - another process may be using it")
            else:
                logger.error(f"Database operational error: {e}")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")

    def hash_password(self, password: str) -> str:
        """Hash a password using SHA256."""
        return hashlib.sha256(password.encode()).hexdigest()

    def generate_token(self) -> str:
        """Generate a secure random token."""
        return secrets.token_urlsafe(32)

    def register_user(self, username: str, password: str, email: Optional[str] = None) -> Dict[str, Any]:
        """Register a new user."""
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.cursor()

                # Check if user exists
                cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
                if cursor.fetchone():
                    return {"success": False, "message": "Username already exists"}

                # Create user
                user_id = secrets.token_urlsafe(16)
                password_hash = self.hash_password(password)

                cursor.execute('''
                    INSERT INTO users (user_id, username, email, password_hash)
                    VALUES (?, ?, ?, ?)
                ''', (user_id, username, email, password_hash))

                conn.commit()

                return {
                    "success": True,
                    "message": "User registered successfully",
                    "user_id": user_id
                }
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return {"success": False, "message": str(e)}

    def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate a user and create a session."""
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.cursor()

                # Get user
                password_hash = self.hash_password(password)
                cursor.execute('''
                    SELECT user_id, username FROM users
                    WHERE username = ? AND password_hash = ?
                ''', (username, password_hash))

                user = cursor.fetchone()
                if not user:
                    return {"success": False, "message": "Invalid credentials"}

                user_id, username = user

                # Create session
                session_token = self.generate_token()
                auth_token = self.generate_token()
                expires_at = datetime.now() + timedelta(hours=24)

                cursor.execute('''
                    INSERT INTO sessions (session_token, user_id, expires_at)
                    VALUES (?, ?, ?)
                ''', (session_token, user_id, expires_at))

                conn.commit()

                return {
                    "success": True,
                    "message": "Login successful",
                    "user_id": user_id,
                    "username": username,
                    "session_token": session_token,
                    "auth_token": auth_token
                }
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return {"success": False, "message": str(e)}

    def validate_session(self, session_token: str) -> Dict[str, Any]:
        """Validate a session token."""
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    SELECT s.user_id, s.expires_at, u.username
                    FROM sessions s
                    JOIN users u ON s.user_id = u.user_id
                    WHERE s.session_token = ?
                ''', (session_token,))

                session = cursor.fetchone()
                if not session:
                    return {"valid": False, "message": "Invalid session"}

                user_id, expires_at, username = session

                # Check expiration
                if datetime.fromisoformat(expires_at) < datetime.now():
                    return {"valid": False, "message": "Session expired"}

                return {
                    "valid": True,
                    "user_id": user_id,
                    "username": username
                }
        except Exception as e:
            logger.error(f"Session validation error: {e}")
            return {"valid": False, "message": str(e)}

    def logout_user(self, session_token: str) -> Dict[str, Any]:
        """Logout user by removing session."""
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM sessions WHERE session_token = ?", (session_token,))
                conn.commit()
                return {"success": True, "message": "Logged out successfully"}
        except Exception as e:
            logger.error(f"Logout error: {e}")
            return {"success": False, "message": str(e)}

    def save_conversation(self, user_id: str, thread_id: str, mode: str, messages: List[Dict]) -> Dict[str, Any]:
        """Save a conversation to database."""
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.cursor()

                conversation_id = secrets.token_urlsafe(16)
                messages_json = json.dumps(messages)

                cursor.execute('''
                    INSERT INTO conversations (conversation_id, user_id, thread_id, mode, messages)
                    VALUES (?, ?, ?, ?, ?)
                ''', (conversation_id, user_id, thread_id, mode, messages_json))

                conn.commit()

                return {"success": True, "conversation_id": conversation_id}
        except Exception as e:
            logger.error(f"Save conversation error: {e}")
            return {"success": False, "message": str(e)}

    def get_user_conversations(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's recent conversations."""
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    SELECT conversation_id, thread_id, mode, messages, created_at
                    FROM conversations
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (user_id, limit))

                conversations = []
                for row in cursor.fetchall():
                    conv_id, thread_id, mode, messages_json, created_at = row
                    conversations.append({
                        "conversation_id": conv_id,
                        "thread_id": thread_id,
                        "mode": mode,
                        "messages": json.loads(messages_json) if messages_json else [],
                        "created_at": created_at
                    })

                return conversations
        except Exception as e:
            logger.error(f"Get conversations error: {e}")
            return []

    def log_user_action(self, user_id: str, action_type: str, details: Optional[Dict] = None) -> Dict[str, Any]:
        """Log a user action."""
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.cursor()

                details_json = json.dumps(details) if details else None

                cursor.execute('''
                    INSERT INTO user_actions (user_id, action_type, details)
                    VALUES (?, ?, ?)
                ''', (user_id, action_type, details_json))

                conn.commit()

                return {"success": True}
        except Exception as e:
            logger.error(f"Log action error: {e}")
            return {"success": False, "message": str(e)}

# Create singleton instance
auth_manager = AuthManager()