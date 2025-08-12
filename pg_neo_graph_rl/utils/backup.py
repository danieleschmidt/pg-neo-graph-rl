"""
Backup and recovery utilities for pg-neo-graph-rl.
Provides checkpointing, model saving, and disaster recovery capabilities.
"""

import os
import json
import pickle
import shutil
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
import threading

import jax
import jax.numpy as jnp
from flax import serialization

from ..utils.exceptions import BackupError, ValidationError
from ..utils.logging import get_logger
from ..utils.security import compute_data_hash

logger = get_logger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint files."""
    checkpoint_id: str
    timestamp: float
    episode: int
    agent_id: Optional[int]
    model_hash: str
    file_path: str
    file_size: int
    description: Optional[str] = None
    tags: List[str] = None


class CheckpointManager:
    """
    Manages model checkpoints and training state backups.
    """
    
    def __init__(self, 
                 base_dir: str = "./checkpoints",
                 max_checkpoints: int = 10,
                 compress: bool = True):
        self.base_dir = Path(base_dir)
        self.max_checkpoints = max_checkpoints
        self.compress = compress
        self.checkpoints: List[CheckpointMetadata] = []
        self._lock = threading.Lock()
        
        # Create checkpoint directory
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing checkpoints
        self._load_checkpoint_index()
        
        logger.info(f"CheckpointManager initialized: {self.base_dir}")
    
    def save_checkpoint(self,
                       state: Dict[str, Any],
                       episode: int,
                       agent_id: Optional[int] = None,
                       description: Optional[str] = None,
                       tags: Optional[List[str]] = None) -> str:
        """
        Save a checkpoint with model state and metadata.
        
        Args:
            state: Model state dictionary
            episode: Training episode number
            agent_id: Agent ID (for multi-agent scenarios)
            description: Optional description
            tags: Optional tags for categorization
            
        Returns:
            Checkpoint ID
        """
        with self._lock:
            checkpoint_id = self._generate_checkpoint_id(episode, agent_id)
            
            # Create checkpoint subdirectory
            checkpoint_dir = self.base_dir / checkpoint_id
            checkpoint_dir.mkdir(exist_ok=True)
            
            try:
                # Save model state
                state_file = checkpoint_dir / "model_state.pkl"
                if self.compress:
                    import gzip
                    with gzip.open(state_file, 'wb') as f:
                        pickle.dump(state, f)
                else:
                    with open(state_file, 'wb') as f:
                        pickle.dump(state, f)
                
                # Compute hash for integrity
                with open(state_file, 'rb') as f:
                    file_data = f.read()
                    model_hash = hashlib.sha256(file_data).hexdigest()
                
                # Create metadata
                metadata = CheckpointMetadata(
                    checkpoint_id=checkpoint_id,
                    timestamp=time.time(),
                    episode=episode,
                    agent_id=agent_id,
                    model_hash=model_hash,
                    file_path=str(state_file),
                    file_size=len(file_data),
                    description=description,
                    tags=tags or []
                )
                
                # Save metadata
                metadata_file = checkpoint_dir / "metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(asdict(metadata), f, indent=2)
                
                # Add to checkpoint list
                self.checkpoints.append(metadata)
                
                # Cleanup old checkpoints
                self._cleanup_old_checkpoints()
                
                # Update index
                self._save_checkpoint_index()
                
                logger.info(f"Checkpoint saved: {checkpoint_id} (episode {episode})")
                return checkpoint_id
                
            except Exception as e:
                # Cleanup on failure
                if checkpoint_dir.exists():
                    shutil.rmtree(checkpoint_dir)
                raise BackupError(f"Failed to save checkpoint: {e}", checkpoint=checkpoint_id)
    
    def load_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Load a checkpoint by ID.
        
        Args:
            checkpoint_id: Checkpoint identifier
            
        Returns:
            Model state dictionary
        """
        with self._lock:
            # Find checkpoint metadata
            metadata = None
            for cp in self.checkpoints:
                if cp.checkpoint_id == checkpoint_id:
                    metadata = cp
                    break
            
            if not metadata:
                raise BackupError(f"Checkpoint not found: {checkpoint_id}")
            
            checkpoint_dir = self.base_dir / checkpoint_id
            state_file = checkpoint_dir / "model_state.pkl"
            
            if not state_file.exists():
                raise BackupError(f"Checkpoint file not found: {state_file}")
            
            try:
                # Load model state
                if self.compress:
                    import gzip
                    with gzip.open(state_file, 'rb') as f:
                        state = pickle.load(f)
                else:
                    with open(state_file, 'rb') as f:
                        state = pickle.load(f)
                
                # Verify integrity
                with open(state_file, 'rb') as f:
                    file_data = f.read()
                    computed_hash = hashlib.sha256(file_data).hexdigest()
                
                if computed_hash != metadata.model_hash:
                    raise BackupError(
                        f"Checkpoint integrity check failed: {checkpoint_id}",
                        checkpoint=checkpoint_id
                    )
                
                logger.info(f"Checkpoint loaded: {checkpoint_id}")
                return state
                
            except Exception as e:
                raise BackupError(f"Failed to load checkpoint: {e}", checkpoint=checkpoint_id)
    
    def list_checkpoints(self, 
                        agent_id: Optional[int] = None,
                        tags: Optional[List[str]] = None) -> List[CheckpointMetadata]:
        """
        List available checkpoints with optional filtering.
        
        Args:
            agent_id: Filter by agent ID
            tags: Filter by tags
            
        Returns:
            List of checkpoint metadata
        """
        with self._lock:
            filtered = self.checkpoints.copy()
            
            if agent_id is not None:
                filtered = [cp for cp in filtered if cp.agent_id == agent_id]
            
            if tags:
                filtered = [cp for cp in filtered if any(tag in (cp.tags or []) for tag in tags)]
            
            # Sort by timestamp (newest first)
            filtered.sort(key=lambda x: x.timestamp, reverse=True)
            
            return filtered
    
    def get_latest_checkpoint(self, agent_id: Optional[int] = None) -> Optional[CheckpointMetadata]:
        """Get the most recent checkpoint."""
        checkpoints = self.list_checkpoints(agent_id=agent_id)
        return checkpoints[0] if checkpoints else None
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint to delete
            
        Returns:
            True if deleted successfully
        """
        with self._lock:
            # Find and remove from list
            self.checkpoints = [cp for cp in self.checkpoints if cp.checkpoint_id != checkpoint_id]
            
            # Remove files
            checkpoint_dir = self.base_dir / checkpoint_id
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)
                logger.info(f"Checkpoint deleted: {checkpoint_id}")
                
                # Update index
                self._save_checkpoint_index()
                return True
            
            return False
    
    def cleanup_checkpoints(self, keep_latest: int = 5) -> int:
        """
        Clean up old checkpoints, keeping only the most recent.
        
        Args:
            keep_latest: Number of checkpoints to keep
            
        Returns:
            Number of checkpoints deleted
        """
        with self._lock:
            if len(self.checkpoints) <= keep_latest:
                return 0
            
            # Sort by timestamp
            sorted_checkpoints = sorted(self.checkpoints, key=lambda x: x.timestamp, reverse=True)
            
            # Keep the latest ones
            to_keep = sorted_checkpoints[:keep_latest]
            to_delete = sorted_checkpoints[keep_latest:]
            
            deleted_count = 0
            for cp in to_delete:
                if self.delete_checkpoint(cp.checkpoint_id):
                    deleted_count += 1
            
            logger.info(f"Cleanup completed: {deleted_count} checkpoints deleted")
            return deleted_count
    
    def _generate_checkpoint_id(self, episode: int, agent_id: Optional[int]) -> str:
        """Generate unique checkpoint ID."""
        timestamp = int(time.time())
        if agent_id is not None:
            return f"checkpoint_ep{episode}_agent{agent_id}_{timestamp}"
        return f"checkpoint_ep{episode}_{timestamp}"
    
    def _cleanup_old_checkpoints(self):
        """Remove oldest checkpoints if we exceed the limit."""
        if len(self.checkpoints) > self.max_checkpoints:
            # Sort by timestamp
            sorted_checkpoints = sorted(self.checkpoints, key=lambda x: x.timestamp)
            
            # Remove oldest
            while len(self.checkpoints) > self.max_checkpoints:
                oldest = sorted_checkpoints.pop(0)
                checkpoint_dir = self.base_dir / oldest.checkpoint_id
                if checkpoint_dir.exists():
                    shutil.rmtree(checkpoint_dir)
                self.checkpoints.remove(oldest)
                logger.info(f"Removed old checkpoint: {oldest.checkpoint_id}")
    
    def _load_checkpoint_index(self):
        """Load checkpoint index from disk."""
        index_file = self.base_dir / "checkpoint_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    data = json.load(f)
                    self.checkpoints = [
                        CheckpointMetadata(**cp_data) for cp_data in data
                    ]
                logger.info(f"Loaded {len(self.checkpoints)} checkpoints from index")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint index: {e}")
                self.checkpoints = []
    
    def _save_checkpoint_index(self):
        """Save checkpoint index to disk."""
        index_file = self.base_dir / "checkpoint_index.json"
        try:
            with open(index_file, 'w') as f:
                json.dump([asdict(cp) for cp in self.checkpoints], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint index: {e}")


class AutoBackup:
    """
    Automatic backup system that runs in the background.
    """
    
    def __init__(self, 
                 checkpoint_manager: CheckpointManager,
                 backup_interval: int = 10,  # episodes
                 auto_cleanup: bool = True):
        self.checkpoint_manager = checkpoint_manager
        self.backup_interval = backup_interval
        self.auto_cleanup = auto_cleanup
        self.last_backup_episode = -1
        self.enabled = True
        
        logger.info(f"AutoBackup initialized: interval={backup_interval} episodes")
    
    def maybe_backup(self, 
                    state: Dict[str, Any],
                    episode: int,
                    agent_id: Optional[int] = None,
                    force: bool = False) -> Optional[str]:
        """
        Backup if conditions are met.
        
        Args:
            state: Model state to backup
            episode: Current episode
            agent_id: Agent ID
            force: Force backup regardless of interval
            
        Returns:
            Checkpoint ID if backup was created
        """
        if not self.enabled:
            return None
        
        should_backup = (
            force or 
            episode % self.backup_interval == 0 or
            episode - self.last_backup_episode >= self.backup_interval
        )
        
        if should_backup:
            try:
                checkpoint_id = self.checkpoint_manager.save_checkpoint(
                    state=state,
                    episode=episode,
                    agent_id=agent_id,
                    description=f"Auto-backup at episode {episode}",
                    tags=["auto-backup"]
                )
                
                self.last_backup_episode = episode
                
                # Auto cleanup if enabled
                if self.auto_cleanup and episode % (self.backup_interval * 5) == 0:
                    self.checkpoint_manager.cleanup_checkpoints(keep_latest=5)
                
                return checkpoint_id
                
            except Exception as e:
                logger.error(f"Auto-backup failed at episode {episode}: {e}")
                return None
        
        return None
    
    def enable(self):
        """Enable automatic backups."""
        self.enabled = True
        logger.info("Auto-backup enabled")
    
    def disable(self):
        """Disable automatic backups."""
        self.enabled = False
        logger.info("Auto-backup disabled")


class DisasterRecovery:
    """
    Disaster recovery utilities for training restoration.
    """
    
    def __init__(self, checkpoint_manager: CheckpointManager):
        self.checkpoint_manager = checkpoint_manager
    
    def create_recovery_plan(self, 
                           target_episode: Optional[int] = None,
                           agent_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Create a recovery plan to restore training to a specific state.
        
        Args:
            target_episode: Target episode to recover to (None for latest)
            agent_id: Specific agent to recover
            
        Returns:
            Recovery plan dictionary
        """
        checkpoints = self.checkpoint_manager.list_checkpoints(agent_id=agent_id)
        
        if not checkpoints:
            raise BackupError("No checkpoints available for recovery")
        
        # Find best checkpoint
        if target_episode is None:
            # Use latest checkpoint
            best_checkpoint = checkpoints[0]
        else:
            # Find checkpoint closest to target episode
            best_checkpoint = min(
                checkpoints,
                key=lambda cp: abs(cp.episode - target_episode)
            )
        
        recovery_plan = {
            "recovery_id": f"recovery_{int(time.time())}",
            "checkpoint_id": best_checkpoint.checkpoint_id,
            "target_episode": target_episode,
            "actual_episode": best_checkpoint.episode,
            "agent_id": agent_id,
            "recovery_timestamp": time.time(),
            "checkpoint_metadata": asdict(best_checkpoint)
        }
        
        logger.info(f"Recovery plan created: {recovery_plan['recovery_id']}")
        return recovery_plan
    
    def execute_recovery(self, recovery_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a recovery plan.
        
        Args:
            recovery_plan: Recovery plan from create_recovery_plan
            
        Returns:
            Recovered state dictionary
        """
        checkpoint_id = recovery_plan["checkpoint_id"]
        
        try:
            # Load checkpoint
            state = self.checkpoint_manager.load_checkpoint(checkpoint_id)
            
            logger.info(f"Recovery executed successfully: {recovery_plan['recovery_id']}")
            return state
            
        except Exception as e:
            raise BackupError(f"Recovery failed: {e}", checkpoint=checkpoint_id)
    
    def validate_recovery_integrity(self, 
                                  recovered_state: Dict[str, Any],
                                  recovery_plan: Dict[str, Any]) -> bool:
        """
        Validate the integrity of recovered state.
        
        Args:
            recovered_state: State returned from execute_recovery
            recovery_plan: Original recovery plan
            
        Returns:
            True if validation passes
        """
        try:
            # Basic structure validation
            required_keys = ["params", "optimizer_state"]  # Common keys
            for key in required_keys:
                if key not in recovered_state:
                    logger.warning(f"Missing key in recovered state: {key}")
                    return False
            
            # Validate parameters are JAX arrays
            if "params" in recovered_state:
                for param_name, param_value in recovered_state["params"].items():
                    if not isinstance(param_value, jnp.ndarray):
                        logger.warning(f"Parameter {param_name} is not a JAX array")
                        return False
            
            logger.info("Recovery integrity validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Recovery validation failed: {e}")
            return False


# Convenience functions
def create_backup_system(checkpoint_dir: str = "./checkpoints") -> tuple:
    """
    Create a complete backup system with all components.
    
    Args:
        checkpoint_dir: Directory for checkpoints
        
    Returns:
        Tuple of (CheckpointManager, AutoBackup, DisasterRecovery)
    """
    checkpoint_manager = CheckpointManager(base_dir=checkpoint_dir)
    auto_backup = AutoBackup(checkpoint_manager)
    disaster_recovery = DisasterRecovery(checkpoint_manager)
    
    logger.info("Complete backup system created")
    return checkpoint_manager, auto_backup, disaster_recovery