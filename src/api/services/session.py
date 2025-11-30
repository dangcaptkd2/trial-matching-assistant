"""Session management service for mapping Chainlit sessions to LangGraph threads."""

from datetime import datetime
from typing import Dict, Optional


class SessionService:
    """Service to manage user sessions and thread mapping."""

    def __init__(self):
        """Initialize session service."""
        # In-memory storage for session metadata
        # In production, this could be stored in Redis or database
        self._sessions: Dict[str, Dict] = {}

    def get_or_create_thread_id(
        self, user_id: str, session_id: Optional[str] = None
    ) -> str:
        """
        Get or create a thread_id for a user session.

        Args:
            user_id: User ID (from Open WebUI or API)
            session_id: Optional session ID (defaults to user_id)

        Returns:
            thread_id: Unique thread identifier for LangGraph
        """
        # Use user_id as thread_id for persistent conversations
        # This ensures the same user always gets the same thread
        if not session_id:
            session_id = user_id

        thread_id = f"user_{user_id}_{session_id}"

        # Store session metadata if not exists
        if thread_id not in self._sessions:
            self._sessions[thread_id] = {
                "user_id": user_id,
                "session_id": session_id,
                "thread_id": thread_id,
                "created_at": datetime.now(),
                "last_activity": datetime.now(),
            }

        # Update last activity
        self._sessions[thread_id]["last_activity"] = datetime.now()

        return thread_id

    def get_session_metadata(self, thread_id: str) -> Optional[Dict]:
        """
        Get session metadata by thread_id.

        Args:
            thread_id: Thread identifier

        Returns:
            Session metadata dict or None if not found
        """
        return self._sessions.get(thread_id)

    def update_last_activity(self, thread_id: str) -> None:
        """
        Update last activity timestamp for a session.

        Args:
            thread_id: Thread identifier
        """
        if thread_id in self._sessions:
            self._sessions[thread_id]["last_activity"] = datetime.now()

    def list_sessions(self, user_id: Optional[str] = None) -> list[Dict]:
        """
        List all sessions, optionally filtered by user_id.

        Args:
            user_id: Optional user ID to filter by

        Returns:
            List of session metadata dicts
        """
        sessions = list(self._sessions.values())
        if user_id:
            sessions = [s for s in sessions if s.get("user_id") == user_id]
        return sessions


# Global session service instance
session_service = SessionService()
