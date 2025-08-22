"""
Custom exceptions for pg-neo-graph-rl.
"""
from typing import Optional


class GraphRLError(Exception):
    """Base exception for all graph RL related errors."""
    pass


class FederatedLearningError(GraphRLError):
    """Raised when federated learning operations fail."""

    def __init__(self, message: str, agent_id: Optional[int] = None, round_number: Optional[int] = None):
        self.agent_id = agent_id
        self.round_number = round_number

        full_message = message
        if agent_id is not None:
            full_message += f" (Agent ID: {agent_id})"
        if round_number is not None:
            full_message += f" (Round: {round_number})"

        super().__init__(full_message)


class NetworkError(GraphRLError):
    """Raised when neural network operations fail."""

    def __init__(self, message: str, network_type: str = None):
        self.network_type = network_type

        full_message = message
        if network_type:
            full_message = f"[{network_type}] {message}"

        super().__init__(full_message)


class EnvironmentError(GraphRLError):
    """Raised when environment operations fail."""

    def __init__(self, message: str, environment_name: str = None, step: int = None):
        self.environment_name = environment_name
        self.step = step

        full_message = message
        if environment_name:
            full_message = f"[{environment_name}] {message}"
        if step is not None:
            full_message += f" (Step: {step})"

        super().__init__(full_message)


class ValidationError(GraphRLError):
    """Raised when input validation fails."""

    def __init__(self, message: str, field_name: str = None, expected_type: str = None):
        self.field_name = field_name
        self.expected_type = expected_type

        full_message = message
        if field_name:
            full_message = f"Field '{field_name}': {message}"
        if expected_type:
            full_message += f" (Expected: {expected_type})"

        super().__init__(full_message)


class CommunicationError(FederatedLearningError):
    """Raised when agent communication fails."""

    def __init__(self, message: str, sender_id: int = None, receiver_id: int = None):
        self.sender_id = sender_id
        self.receiver_id = receiver_id

        full_message = message
        if sender_id is not None and receiver_id is not None:
            full_message += f" (From Agent {sender_id} to Agent {receiver_id})"

        super().__init__(full_message)


class SecurityError(GraphRLError):
    """Raised when security violations are detected."""

    def __init__(self, message: str, security_level: str = "HIGH"):
        self.security_level = security_level
        full_message = f"[SECURITY-{security_level}] {message}"
        super().__init__(full_message)


class ResourceError(GraphRLError):
    """Raised when system resources are insufficient."""

    def __init__(self, message: str, resource_type: str = None):
        self.resource_type = resource_type

        full_message = message
        if resource_type:
            full_message = f"[{resource_type}] {message}"

        super().__init__(full_message)


class TrainingError(GraphRLError):
    """Raised when training operations fail."""

    def __init__(self, message: str, episode: int = None, agent_id: int = None):
        self.episode = episode
        self.agent_id = agent_id

        full_message = message
        if episode is not None:
            full_message += f" (Episode: {episode})"
        if agent_id is not None:
            full_message += f" (Agent: {agent_id})"

        super().__init__(full_message)


class CircuitBreakerError(GraphRLError):
    """Raised when circuit breaker is open."""

    def __init__(self, message: str, service_name: str = None, failure_count: int = None):
        self.service_name = service_name
        self.failure_count = failure_count

        full_message = message
        if service_name:
            full_message = f"[{service_name}] {message}"
        if failure_count is not None:
            full_message += f" (Failures: {failure_count})"

        super().__init__(full_message)


class ConfigurationError(GraphRLError):
    """Raised when configuration is invalid."""

    def __init__(self, message: str, config_key: str = None, config_file: str = None):
        self.config_key = config_key
        self.config_file = config_file

        full_message = message
        if config_key:
            full_message += f" (Key: {config_key})"
        if config_file:
            full_message += f" (File: {config_file})"

        super().__init__(full_message)


class TimeoutError(GraphRLError):
    """Raised when operations timeout."""

    def __init__(self, message: str, timeout_seconds: float = None, operation: str = None):
        self.timeout_seconds = timeout_seconds
        self.operation = operation

        full_message = message
        if operation:
            full_message = f"[{operation}] {message}"
        if timeout_seconds is not None:
            full_message += f" (Timeout: {timeout_seconds}s)"

        super().__init__(full_message)


class BackupError(GraphRLError):
    """Raised when backup/recovery operations fail."""

    def __init__(self, message: str, backup_type: str = None, checkpoint: str = None):
        self.backup_type = backup_type
        self.checkpoint = checkpoint

        full_message = message
        if backup_type:
            full_message = f"[{backup_type}] {message}"
        if checkpoint:
            full_message += f" (Checkpoint: {checkpoint})"

        super().__init__(full_message)


class DistributedComputeError(GraphRLError):
    """Raised when distributed computing operations fail."""

    def __init__(self, message: str, node_id: str = None, task_id: str = None):
        self.node_id = node_id
        self.task_id = task_id

        full_message = message
        if node_id:
            full_message = f"[Node {node_id}] {message}"
        if task_id:
            full_message += f" (Task: {task_id})"

        super().__init__(full_message)
