"""
ChatSessionManager for maintaining persistent ChatSession instances across requests.
Provides conversation continuity for AI assistant interactions.
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Any
from app2.llm.chat_wrapper import ChatSession
from app2.llm import available_models
from app2.llm.tools.edit_tools import EDIT_TOOLS

# Set up logger
logger = logging.getLogger("beatgen.chat_session_manager")


class ChatSessionManager:
    """
    Manager for persistent ChatSession instances.
    Maintains one ChatSession per user+mode combination with automatic cleanup.
    """

    # Singleton instance
    _instance = None

    # Class constants
    SESSION_TIMEOUT_SECONDS = 300  # 5 minutes of inactivity
    CLEANUP_INTERVAL_SECONDS = 60  # 1 minute
    MAX_SESSIONS = 100  # Prevent memory exhaustion

    def __new__(cls):
        """Singleton pattern to ensure only one chat session manager exists"""
        if cls._instance is None:
            cls._instance = super(ChatSessionManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the chat session manager"""
        # Skip initialization if already done
        if self._initialized:
            return

        # Active sessions by project_id:mode
        self._sessions: Dict[str, Dict[str, Any]] = {}

        # Initialize cleanup task as None
        self._cleanup_task = None

        self._initialized = True
        logger.info("ChatSessionManager initialized")

    def _get_session_key(self, project_id: str, mode: str) -> str:
        """Generate a unique key for project+mode combination"""
        return f"{project_id}:{mode}"

    def _ensure_cleanup_task(self):
        """Ensure the cleanup task is running"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Started ChatSessionManager cleanup task")

    async def _cleanup_loop(self):
        """Background task to clean up expired sessions"""
        while True:
            try:
                await asyncio.sleep(self.CLEANUP_INTERVAL_SECONDS)
                self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                logger.info("ChatSessionManager cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in ChatSessionManager cleanup: {e}")

    def _cleanup_expired_sessions(self):
        """Remove expired sessions based on last activity"""
        current_time = time.time()
        expired_keys = []

        for session_key, session_data in self._sessions.items():
            last_activity = session_data.get('last_activity', 0)
            if current_time - last_activity > self.SESSION_TIMEOUT_SECONDS:
                expired_keys.append(session_key)

        for key in expired_keys:
            logger.info(f"Cleaning up expired ChatSession: {key}")
            del self._sessions[key]

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired chat sessions")

        # Force cleanup if we have too many sessions
        if len(self._sessions) > self.MAX_SESSIONS:
            logger.warning(f"Too many sessions ({len(self._sessions)}), forcing cleanup")
            # Remove oldest sessions
            sorted_sessions = sorted(
                self._sessions.items(),
                key=lambda x: x[1].get('last_activity', 0)
            )
            sessions_to_remove = len(self._sessions) - self.MAX_SESSIONS + 10
            for i in range(sessions_to_remove):
                if i < len(sorted_sessions):
                    key = sorted_sessions[i][0]
                    logger.info(f"Force removing old session: {key}")
                    del self._sessions[key]

    def get_or_create_session(
        self,
        project_id: str,
        mode: str,
        model: str,
        sse_queue: Any,
        user_id: Optional[str] = None,
        track_context: Optional[Dict[str, Any]] = None,
        db_session: Optional[Any] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> ChatSession:
        """
        Get existing ChatSession or create a new one for project+mode combination.

        Args:
            project_id: Project ID for session isolation (each project has its own conversation)
            mode: Mode (edit, chat, etc.) for context isolation
            model: Model name to use
            sse_queue: SSE queue for streaming
            user_id: User ID (for logging/debugging purposes)
            track_context: Track context for edit mode
            db_session: Database session for saving assistant messages
            request_id: Request ID for message metadata
            **kwargs: Additional arguments for ChatSession

        Returns:
            ChatSession instance
        """
        # Ensure cleanup task is running
        self._ensure_cleanup_task()

        session_key = self._get_session_key(project_id, mode)
        current_time = time.time()

        # Check if we have an existing session
        if session_key in self._sessions:
            session_data = self._sessions[session_key]
            chat_session = session_data['session']
            
            # Update last activity
            session_data['last_activity'] = current_time
            
            # Update the queue for the session (it changes per request)
            chat_session.queue = sse_queue
            
            # Update database context (changes per request)
            chat_session.db_session = db_session
            chat_session.request_id = request_id
            chat_session.mode = mode
            
            # Update track context in system prompt if it changed
            if mode == "edit" and track_context:
                new_system_prompt = self._build_edit_system_prompt(track_context)
                if chat_session.current_system_prompt != new_system_prompt:
                    logger.info(f"Updating system prompt for session: {session_key}")
                    chat_session.current_system_prompt = new_system_prompt
            
            logger.info(f"Reusing existing ChatSession: {session_key} (project: {project_id})")
            return chat_session

        # Create new session
        logger.info(f"Creating new ChatSession: {session_key} with model: {model} (project: {project_id})")

        # Get model info
        model_info = available_models.get_model_by_name(model) if model else available_models.ALL_MODELS[0]

        # Build system prompt based on mode
        if mode == "edit":
            system_prompt = self._build_edit_system_prompt(track_context)
            tools = EDIT_TOOLS
        else:
            # For other modes, use a generic system prompt
            system_prompt = "You are a helpful AI music production assistant."
            tools = None

        # Create ChatSession
        chat_session = ChatSession(
            provider_name=model_info.provider_name,
            model_name=model_info.model_name,
            queue=sse_queue,
            system_prompt=system_prompt,
            api_key=model_info.get_api_key(),
            base_url=getattr(model_info, 'base_url', None),
            tools=tools,
            project_id=project_id,
            db_session=db_session,
            request_id=request_id,
            mode=mode,
            **getattr(model_info, 'provider_kwargs', {}),
            **kwargs
        )

        # Store session with metadata
        self._sessions[session_key] = {
            'session': chat_session,
            'project_id': project_id,
            'user_id': user_id,
            'mode': mode,
            'model': model,
            'created_at': current_time,
            'last_activity': current_time,
        }

        logger.info(f"Created ChatSession: {session_key}, total sessions: {len(self._sessions)}")
        return chat_session

    def _build_edit_system_prompt(self, track_context: Optional[Dict[str, Any]] = None) -> str:
        """Build system prompt for edit mode with track context"""
        
        # Extract comprehensive track information from context
        selected_track_id = None
        all_tracks = []
        project_info = {}
        
        if track_context:
            # Try multiple sources for track ID (single track)
            selected_track_id = (
                track_context.get('track_id') or 
                track_context.get('id')
            )
            
            # Get project-level information
            project_info = {
                'name': track_context.get('project_name') or track_context.get('name'),
                'bpm': track_context.get('bpm'),
                'key': track_context.get('key') or track_context.get('key_signature'),
                'time_signature': track_context.get('time_signature')
            }
            
            # Check for comprehensive tracks in context
            if track_context.get('tracks'):
                for track in track_context['tracks']:
                    track_info = {
                        'id': track.get('id', 'unknown'),
                        'name': track.get('name', 'Unnamed'),
                        'type': track.get('type', 'unknown'),
                        'volume': track.get('volume', 1.0),
                        'pan': track.get('pan', 0.0),
                        'muted': track.get('muted', False),
                        'soloed': track.get('soloed', False),
                        'instrument': track.get('instrument') or track.get('instrument_name')
                    }
                    all_tracks.append(track_info)

        # Build comprehensive track context info
        track_context_info = ""
        if all_tracks:
            track_lines = []
            for track in all_tracks:
                status_parts = []
                if track['muted']:
                    status_parts.append("MUTED")
                if track['soloed']:
                    status_parts.append("SOLOED")
                
                status_str = f" ({', '.join(status_parts)})" if status_parts else ""
                volume_str = f" Vol:{track['volume']:.1%}" if track['volume'] != 1.0 else ""
                pan_str = f" Pan:{track['pan']:+.1f}" if track['pan'] != 0.0 else ""
                
                track_line = f"- {track['name']} (ID: {track['id']}, Type: {track['type'].upper()})"
                if track['instrument']:
                    track_line += f", Instrument: {track['instrument']}"
                track_line += f"{volume_str}{pan_str}{status_str}"
                track_lines.append(track_line)
            
            track_context_info = f"\n\nPROJECT TRACKS:\n" + "\n".join(track_lines)
            
        if project_info.get('name') or project_info.get('bpm'):
            project_parts = []
            if project_info.get('name'):
                project_parts.append(f"Name: {project_info['name']}")
            if project_info.get('bpm'):
                project_parts.append(f"BPM: {project_info['bpm']}")
            if project_info.get('key'):
                project_parts.append(f"Key: {project_info['key']}")
            if project_info.get('time_signature'):
                ts = project_info['time_signature']
                if isinstance(ts, dict):
                    project_parts.append(f"Time Signature: {ts.get('numerator', 4)}/{ts.get('denominator', 4)}")
                else:
                    project_parts.append(f"Time Signature: {ts}")
            
            track_context_info = f"\nPROJECT INFO: {', '.join(project_parts)}" + track_context_info
            
        if selected_track_id:
            track_context_info += f"\n\nCURRENTLY SELECTED TRACK: {selected_track_id}"
            
        if all_tracks or selected_track_id or project_info:
            return f"""You are a music production assistant specializing in editing tracks and projects.

Your task is to understand the user's edit request and call the appropriate tool to make the change.

Available tools:
- adjust_volume: Change track volume levels (requires track_id)
- adjust_pan: Change track stereo panning (requires track_id)
- toggle_mute: Mute or unmute tracks (requires track_id)
- toggle_solo: Solo or unsolo tracks (requires track_id)
- change_bpm: Change project tempo
- change_key: Change project key signature
- change_time_signature: Change project time signature  
- change_project_name: Change project name
{track_context_info}

IMPORTANT: For track-specific operations (volume, pan, mute, solo):
- When user refers to tracks by NAME, use the corresponding track ID from the PROJECT TRACKS list above
- If user says "mute the piano track" or "turn up the drums", find the track name and use its ID
- If user refers to "this track" or "the track", use: {selected_track_id or "the selected track ID"}
- You can reference tracks by their names, types, or instruments - always use the track ID for tool calls

TRACK NAME TO ID MAPPING:
{chr(10).join([f"- '{track['name']}' → {track['id']}" for track in all_tracks]) if all_tracks else "No tracks available"}

You can call MULTIPLE tools in a single response if needed:
- If user says "mute the piano and drums", call toggle_mute multiple times with different track IDs
- If user says "set the bass volume to 50% and pan left", call adjust_volume then adjust_pan  
- If user wants multiple changes, make all the necessary tool calls

Call the appropriate tool(s) based on what the user wants to edit. Be precise with parameter values.

Remember our conversation history - you can reference previous edits and build upon them."""
        else:
            return """You are a music production assistant specializing in editing tracks and projects.

Your task is to understand the user's edit request and call the appropriate tool to make the change.

Available tools:
- adjust_volume: Change track volume levels (requires track_id)
- adjust_pan: Change track stereo panning (requires track_id)  
- toggle_mute: Mute or unmute tracks (requires track_id)
- toggle_solo: Solo or unsolo tracks (requires track_id)
- change_bpm: Change project tempo
- change_key: Change project key signature
- change_time_signature: Change project time signature
- change_project_name: Change project name

WARNING: No track context is available. For track-specific operations (volume, pan, mute, solo), 
you need to ask the user to select or specify which track they want to edit.
For project-level operations (BPM, key, time signature, name), you can proceed normally.

You can call MULTIPLE tools in a single response if needed:
- For multiple project changes (BPM + key), call multiple tools
- If user requests several changes, make all the necessary tool calls

Remember our conversation history - you can reference previous edits and build upon them."""

    def invalidate_session(self, project_id: str, mode: str):
        """Remove a specific session (useful for testing or manual cleanup)"""
        session_key = self._get_session_key(project_id, mode)
        if session_key in self._sessions:
            logger.info(f"Invalidating ChatSession: {session_key}")
            del self._sessions[session_key]

    def get_session_count(self) -> int:
        """Get the current number of active sessions"""
        return len(self._sessions)

    def get_session_info(self) -> Dict[str, Any]:
        """Get information about all active sessions (for debugging)"""
        current_time = time.time()
        info = {}
        
        for session_key, session_data in self._sessions.items():
            info[session_key] = {
                'project_id': session_data['project_id'],
                'user_id': session_data.get('user_id'),
                'mode': session_data['mode'],
                'model': session_data['model'],
                'created_at': session_data['created_at'],
                'last_activity': session_data['last_activity'],
                'age_seconds': current_time - session_data['created_at'],
                'inactive_seconds': current_time - session_data['last_activity'],
            }
        
        return info


# Global singleton instance
chat_session_manager = ChatSessionManager()