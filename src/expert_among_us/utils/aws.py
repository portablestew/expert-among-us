"""AWS utilities for credential management and MCP compatibility.

This module provides utilities to handle AWS credential operations with
automatic refresh for long-running processes.
"""

import boto3
from datetime import datetime, timezone
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Optional


class AWSCredentialError(Exception):
    """Raised when AWS credentials are missing, invalid, or cannot be refreshed."""
    pass


class AWSSessionManager:
    """Manages AWS session with automatic credential refresh.
    
    Monitors credential expiration and automatically refreshes the session
    when credentials are about to expire, enabling long-running processes
    to pick up externally updated credentials from ~/.aws/credentials.
    
    Example:
        aws = AWSSessionManager("us-west-2")
        bedrock = aws.get_client("bedrock-runtime")
        response = bedrock.converse(...)  # Credentials auto-refresh as needed
    """
    
    def __init__(self, region_name: str = "us-west-2", refresh_buffer_seconds: int = 300, session_ttl_seconds: int = 900):
        """Initialize session manager.
        
        Args:
            region_name: AWS region (default: us-west-2)
            refresh_buffer_seconds: Seconds before expiry to refresh (default: 300 = 5 min)
            session_ttl_seconds: Max session age before refresh (default: 900 = 15 min)
            
        Raises:
            AWSCredentialError: If credentials are missing or invalid
        """
        self.region_name = region_name
        self.refresh_buffer_seconds = refresh_buffer_seconds
        self.session_ttl_seconds = session_ttl_seconds
        self._session: Optional[boto3.Session] = None
        self._session_created_at: Optional[datetime] = None
        self._refresh_session()
    
    def _refresh_session(self):
        """Create a new session with current credentials from disk."""
        session = boto3.Session(region_name=self.region_name)
        
        try:
            # Force credential resolution and validation
            sts = session.client('sts')
            sts.get_caller_identity()
        except NoCredentialsError as e:
            raise AWSCredentialError(
                f"AWS credentials not found. Configure AWS CLI or set environment variables "
                f"(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY).\n{e}"
            )
        except ClientError as e:
            raise AWSCredentialError(f"Invalid AWS credentials: {e}")
        
        self._session = session
        self._session_created_at = datetime.now(timezone.utc)
    
    def _credentials_expired(self) -> bool:
        """Check if current credentials are expired or about to expire."""
        if not self._session:
            return True
        
        try:
            credentials = self._session.get_credentials()
            
            # Check expiry time for temporary credentials
            if hasattr(credentials, '_expiry_time') and credentials._expiry_time:
                expiry = credentials._expiry_time
                now = datetime.now(timezone.utc)
                # Refresh if expiring within buffer period
                return (expiry - now).total_seconds() < self.refresh_buffer_seconds
            # Fallback: check session age for file-based credentials
            elif self._session_created_at:
                age_seconds = (datetime.now(timezone.utc) - self._session_created_at).total_seconds()
                return age_seconds >= self.session_ttl_seconds
            
            # Static credentials never expire
            return False
            
        except Exception:
            # If we can't check, assume expired to be safe
            return True
    
    def ensure_fresh_session(self):
        """Refresh session if credentials are expired or about to expire."""
        if self._credentials_expired():
            self._refresh_session()
    
    def get_client(self, service_name: str, **kwargs):
        """Get an AWS client with fresh credentials.
        
        Args:
            service_name: AWS service name (e.g., 'bedrock-runtime', 'sts')
            **kwargs: Additional arguments passed to client creation
            
        Returns:
            boto3 client with current credentials
        """
        self.ensure_fresh_session()
        return self._session.client(service_name, **kwargs)