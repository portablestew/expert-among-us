"""Debug logging infrastructure for Bedrock API interactions."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import threading
import uuid


class DebugLogger:
    """Debug logger for Bedrock requests and responses (singleton pattern)."""
    
    _enabled: bool = False
    _log_dir: Optional[Path] = None
    _lock = threading.Lock()
    
    @classmethod
    def configure(cls, enabled: bool = False, log_dir: Optional[Path] = None) -> None:
        """Configure the debug logger.
        
        Args:
            enabled: Whether debug logging is enabled
            log_dir: Directory for log files (default: ~/.expert-among-us/logs)
        """
        with cls._lock:
            cls._enabled = enabled
            cls._log_dir = log_dir or Path.home() / ".expert-among-us" / "logs"
            
            if cls._enabled and cls._log_dir:
                cls._log_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def is_enabled(cls) -> bool:
        """Check if debug mode is enabled.
        
        Returns:
            True if debug logging is enabled
        """
        return cls._enabled
    
    @classmethod
    def log_request(cls, operation: str, payload: Dict[str, Any], request_id: Optional[str] = None, category: str = "general") -> str:
        """Log request to JSON file with timestamp.
        
        Args:
            operation: Name of the operation (e.g., "converse", "embed")
            payload: Request payload to log
            request_id: Optional request ID to use. If not provided, generates a new UUID.
            category: Request category for subdirectory (e.g., "embedding", "promptgen", "expert")
            
        Returns:
            The request_id used for this request
        """
        if not cls._enabled:
            return request_id or str(uuid.uuid4())
        
        # Generate request_id if not provided
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        cls._log("request", operation, payload, request_id, category)
        return request_id
    
    @classmethod
    def log_response(cls, operation: str, payload: Dict[str, Any], request_id: Optional[str] = None, category: str = "general") -> None:
        """Log response to JSON file with timestamp.
        
        Args:
            operation: Name of the operation (e.g., "converse", "embed")
            payload: Response payload to log
            request_id: Optional request ID to match with corresponding request
            category: Request category for subdirectory (e.g., "embedding", "promptgen", "expert")
        """
        if not cls._enabled:
            return
        
        # Generate request_id if not provided (for backward compatibility)
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        cls._log("response", operation, payload, request_id, category)
    
    @classmethod
    def _log(cls, log_type: str, operation: str, data: Dict[str, Any], request_id: str, category: str = "general") -> None:
        """Internal method to write log file.
        
        Args:
            log_type: Type of log ("request" or "response")
            operation: Operation name
            data: Data to log
            request_id: Request ID to include in filename
            category: Request category for subdirectory (e.g., "embedding", "promptgen", "expert")
        """
        if not cls._enabled or not cls._log_dir:
            return
        
        # Create category subdirectory
        category_dir = cls._log_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate compact timestamp (e.g., 20251029T054015Z)
        now = datetime.now(timezone.utc)
        timestamp = now.strftime('%Y%m%dT%H%M%SZ')
        # Truncate request_id to first 4 characters for brevity
        short_request_id = request_id[:4]
        filename = f"{operation}_{timestamp}_{short_request_id}_{log_type}.json"
        filepath = category_dir / filename
        
        # Prepare log entry with metadata
        log_entry = {
            "timestamp": now.isoformat(),
            "type": log_type,
            "operation": operation,
            "payload": data
        }
        
        # Write with thread safety
        with cls._lock:
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(log_entry, f, indent=2, default=cls._json_serializer)
            except Exception:
                # Silent failure - don't break the application for logging issues
                pass
    
    @staticmethod
    def _json_serializer(obj: Any) -> str:
        """Custom JSON serializer for datetime and other objects.
        
        Args:
            obj: Object to serialize
            
        Returns:
            String representation of the object
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)