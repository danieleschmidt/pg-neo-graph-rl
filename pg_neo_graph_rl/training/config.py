"""
Training configuration for federated graph reinforcement learning.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class TrainingConfig:
    """Configuration for federated training."""

    # Environment settings
    environment_type: str = "traffic"
    environment_params: Dict[str, Any] = None

    # Federated learning settings
    num_agents: int = 10
    aggregation_method: str = "gossip"
    communication_rounds: int = 10
    topology: str = "random"
    privacy_noise: float = 0.0

    # Algorithm settings
    algorithm: str = "ppo"
    learning_rate: float = 3e-4
    batch_size: int = 64
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5

    # Training settings
    total_episodes: int = 1000
    max_episode_length: int = 200
    eval_frequency: int = 50
    save_frequency: int = 100

    # Optimization settings
    use_jax_compilation: bool = True
    parallel_envs: int = 1

    # Monitoring settings
    enable_monitoring: bool = True
    log_frequency: int = 10
    metrics_port: int = 8080

    # Output settings
    output_dir: Path = Path("./outputs")
    checkpoint_dir: Path = Path("./checkpoints")
    log_dir: Path = Path("./logs")

    def __post_init__(self):
        """Post-initialization setup."""
        if self.environment_params is None:
            self.environment_params = {}

        # Ensure directories are Path objects
        self.output_dir = Path(self.output_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.log_dir = Path(self.log_dir)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "TrainingConfig":
        """Load configuration from YAML file."""
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def to_yaml(self, yaml_path: Path) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            "environment_type": self.environment_type,
            "environment_params": self.environment_params,
            "num_agents": self.num_agents,
            "aggregation_method": self.aggregation_method,
            "communication_rounds": self.communication_rounds,
            "topology": self.topology,
            "privacy_noise": self.privacy_noise,
            "algorithm": self.algorithm,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "clip_epsilon": self.clip_epsilon,
            "entropy_coeff": self.entropy_coeff,
            "value_coeff": self.value_coeff,
            "total_episodes": self.total_episodes,
            "max_episode_length": self.max_episode_length,
            "eval_frequency": self.eval_frequency,
            "save_frequency": self.save_frequency,
            "use_jax_compilation": self.use_jax_compilation,
            "parallel_envs": self.parallel_envs,
            "enable_monitoring": self.enable_monitoring,
            "log_frequency": self.log_frequency,
            "metrics_port": self.metrics_port,
            "output_dir": str(self.output_dir),
            "checkpoint_dir": str(self.checkpoint_dir),
            "log_dir": str(self.log_dir)
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def create_directories(self) -> None:
        """Create necessary directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
