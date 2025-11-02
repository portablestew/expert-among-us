"""Custom exceptions for Expert Among Us API."""


class ExpertAPIError(Exception):
    """Base exception for all Expert Among Us API errors."""
    pass


class ExpertNotFoundError(ExpertAPIError):
    """Raised when an expert does not exist."""
    
    def __init__(self, expert_name: str, message: str = None):
        """Initialize exception.
        
        Args:
            expert_name: Name of the expert that was not found
            message: Optional custom message
        """
        self.expert_name = expert_name
        if message is None:
            message = (
                f"Expert '{expert_name}' does not exist. "
                f"Please run 'populate' command first to create the expert index."
            )
        super().__init__(message)


class ExpertAlreadyExistsError(ExpertAPIError):
    """Raised when attempting to create an expert that already exists."""
    
    def __init__(self, expert_name: str, message: str = None):
        """Initialize exception.
        
        Args:
            expert_name: Name of the expert that already exists
            message: Optional custom message
        """
        self.expert_name = expert_name
        if message is None:
            message = f"Expert '{expert_name}' already exists in data directory."
        super().__init__(message)


class InvalidExpertError(ExpertAPIError):
    """Raised when an expert directory is invalid or corrupted."""
    
    def __init__(self, expert_name: str, reason: str, message: str = None):
        """Initialize exception.
        
        Args:
            expert_name: Name of the invalid expert
            reason: Reason why the expert is invalid
            message: Optional custom message
        """
        self.expert_name = expert_name
        self.reason = reason
        if message is None:
            message = f"Invalid expert '{expert_name}': {reason}"
        super().__init__(message)


class NoResultsError(ExpertAPIError):
    """Raised when a query returns no results."""
    
    def __init__(self, message: str = "No relevant examples found"):
        """Initialize exception.
        
        Args:
            message: Optional custom message
        """
        super().__init__(message)