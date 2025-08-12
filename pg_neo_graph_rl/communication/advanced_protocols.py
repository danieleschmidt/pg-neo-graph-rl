"""
Advanced communication protocols for optimized federated learning.
Includes compression, efficient aggregation, and bandwidth optimization.
"""

import asyncio
import time
import zlib
import threading
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np

import jax
import jax.numpy as jnp
from flax import serialization

from ..utils.exceptions import CommunicationError, GraphRLError
from ..utils.logging import get_logger
from ..utils.fault_tolerance import retry_with_backoff, RetryConfig

logger = get_logger(__name__)


class CompressionType(Enum):
    """Compression algorithms for model parameters."""
    NONE = "none"
    GZIP = "gzip"
    QUANTIZATION = "quantization"
    SPARSIFICATION = "sparsification"
    LOW_RANK = "low_rank"
    ADAPTIVE = "adaptive"


class AggregationStrategy(Enum):
    """Aggregation strategies for federated learning."""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    SCAFFOLD = "scaffold"
    FEDNOVA = "fednova"
    TRIMMED_MEAN = "trimmed_mean"
    MEDIAN = "median"
    KRUM = "krum"
    BULYAN = "bulyan"


@dataclass
class CompressionConfig:
    """Configuration for parameter compression."""
    compression_type: CompressionType
    compression_ratio: float = 0.1  # Target compression ratio
    quantization_bits: int = 8
    sparsity_ratio: float = 0.9  # For sparsification
    low_rank_ratio: float = 0.5  # For low-rank approximation
    adaptive_threshold: float = 0.001  # For adaptive compression


@dataclass
class CommunicationMetrics:
    """Metrics for communication performance."""
    bytes_sent: int = 0
    bytes_received: int = 0
    compression_ratio: float = 1.0
    latency_ms: float = 0.0
    bandwidth_mbps: float = 0.0
    error_count: int = 0
    timestamp: float = field(default_factory=time.time)


class ParameterCompressor:
    """
    Advanced parameter compression for efficient communication.
    """
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.logger = get_logger("pg_neo_graph_rl.compressor")
        
        # Compression statistics
        self.original_sizes: List[int] = []
        self.compressed_sizes: List[int] = []
        
    def compress(self, parameters: Dict[str, jnp.ndarray]) -> Tuple[bytes, Dict[str, Any]]:
        """
        Compress model parameters.
        
        Args:
            parameters: Dictionary of parameter arrays
            
        Returns:
            Tuple of (compressed_data, metadata)
        """
        start_time = time.time()
        
        if self.config.compression_type == CompressionType.NONE:
            return self._no_compression(parameters)
        
        elif self.config.compression_type == CompressionType.GZIP:
            return self._gzip_compression(parameters)
        
        elif self.config.compression_type == CompressionType.QUANTIZATION:
            return self._quantization_compression(parameters)
        
        elif self.config.compression_type == CompressionType.SPARSIFICATION:
            return self._sparsification_compression(parameters)
        
        elif self.config.compression_type == CompressionType.LOW_RANK:
            return self._low_rank_compression(parameters)
        
        elif self.config.compression_type == CompressionType.ADAPTIVE:
            return self._adaptive_compression(parameters)
        
        else:
            raise ValueError(f"Unknown compression type: {self.config.compression_type}")
    
    def decompress(self, compressed_data: bytes, metadata: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
        """
        Decompress model parameters.
        
        Args:
            compressed_data: Compressed parameter data
            metadata: Compression metadata
            
        Returns:
            Dictionary of parameter arrays
        """
        compression_type = CompressionType(metadata.get("compression_type", "none"))
        
        if compression_type == CompressionType.NONE:
            return self._no_decompression(compressed_data, metadata)
        
        elif compression_type == CompressionType.GZIP:
            return self._gzip_decompression(compressed_data, metadata)
        
        elif compression_type == CompressionType.QUANTIZATION:
            return self._quantization_decompression(compressed_data, metadata)
        
        elif compression_type == CompressionType.SPARSIFICATION:
            return self._sparsification_decompression(compressed_data, metadata)
        
        elif compression_type == CompressionType.LOW_RANK:
            return self._low_rank_decompression(compressed_data, metadata)
        
        elif compression_type == CompressionType.ADAPTIVE:
            return self._adaptive_decompression(compressed_data, metadata)
        
        else:
            raise ValueError(f"Unknown compression type: {compression_type}")
    
    def _no_compression(self, parameters: Dict[str, jnp.ndarray]) -> Tuple[bytes, Dict[str, Any]]:
        """No compression - direct serialization."""
        data = serialization.to_bytes(parameters)
        metadata = {
            "compression_type": CompressionType.NONE.value,
            "original_size": len(data)
        }
        return data, metadata
    
    def _no_decompression(self, data: bytes, metadata: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
        """No decompression - direct deserialization."""
        return serialization.from_bytes(None, data)
    
    def _gzip_compression(self, parameters: Dict[str, jnp.ndarray]) -> Tuple[bytes, Dict[str, Any]]:
        """GZIP compression."""
        original_data = serialization.to_bytes(parameters)
        compressed_data = zlib.compress(original_data, level=9)
        
        metadata = {
            "compression_type": CompressionType.GZIP.value,
            "original_size": len(original_data),
            "compressed_size": len(compressed_data)
        }
        
        return compressed_data, metadata
    
    def _gzip_decompression(self, data: bytes, metadata: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
        """GZIP decompression."""
        decompressed_data = zlib.decompress(data)
        return serialization.from_bytes(None, decompressed_data)
    
    def _quantization_compression(self, parameters: Dict[str, jnp.ndarray]) -> Tuple[bytes, Dict[str, Any]]:
        """Quantization-based compression."""
        quantized_params = {}
        scales = {}
        zeros = {}
        
        for name, param in parameters.items():
            # Calculate quantization scale and zero point
            param_min = jnp.min(param)
            param_max = jnp.max(param)
            
            scale = (param_max - param_min) / (2 ** self.config.quantization_bits - 1)
            zero_point = param_min
            
            # Quantize
            quantized = jnp.round((param - zero_point) / scale).astype(jnp.uint8)
            
            quantized_params[name] = quantized
            scales[name] = scale
            zeros[name] = zero_point
        
        # Serialize quantized data
        data_to_compress = {
            "quantized": quantized_params,
            "scales": scales,
            "zeros": zeros
        }
        
        serialized = serialization.to_bytes(data_to_compress)
        compressed = zlib.compress(serialized)
        
        metadata = {
            "compression_type": CompressionType.QUANTIZATION.value,
            "quantization_bits": self.config.quantization_bits,
            "original_size": sum(param.nbytes for param in parameters.values()),
            "compressed_size": len(compressed)
        }
        
        return compressed, metadata
    
    def _quantization_decompression(self, data: bytes, metadata: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
        """Quantization decompression."""
        decompressed = zlib.decompress(data)
        data_dict = serialization.from_bytes(None, decompressed)
        
        parameters = {}
        for name, quantized in data_dict["quantized"].items():
            scale = data_dict["scales"][name]
            zero_point = data_dict["zeros"][name]
            
            # Dequantize
            dequantized = quantized.astype(jnp.float32) * scale + zero_point
            parameters[name] = dequantized
        
        return parameters
    
    def _sparsification_compression(self, parameters: Dict[str, jnp.ndarray]) -> Tuple[bytes, Dict[str, Any]]:
        """Sparsification-based compression."""
        sparse_params = {}
        
        for name, param in parameters.items():
            # Find threshold for sparsification
            flat_param = param.flatten()
            threshold_idx = int(len(flat_param) * self.config.sparsity_ratio)
            sorted_abs = jnp.sort(jnp.abs(flat_param))
            threshold = sorted_abs[threshold_idx] if threshold_idx < len(sorted_abs) else 0
            
            # Create sparse representation
            mask = jnp.abs(param) > threshold
            indices = jnp.where(mask)
            values = param[indices]
            
            sparse_params[name] = {
                "indices": indices,
                "values": values,
                "shape": param.shape,
                "threshold": threshold
            }
        
        serialized = serialization.to_bytes(sparse_params)
        compressed = zlib.compress(serialized)
        
        metadata = {
            "compression_type": CompressionType.SPARSIFICATION.value,
            "sparsity_ratio": self.config.sparsity_ratio,
            "original_size": sum(param.nbytes for param in parameters.values()),
            "compressed_size": len(compressed)
        }
        
        return compressed, metadata
    
    def _sparsification_decompression(self, data: bytes, metadata: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
        """Sparsification decompression."""
        decompressed = zlib.decompress(data)
        sparse_dict = serialization.from_bytes(None, decompressed)
        
        parameters = {}
        for name, sparse_data in sparse_dict.items():
            # Reconstruct dense parameter
            param = jnp.zeros(sparse_data["shape"])
            param = param.at[sparse_data["indices"]].set(sparse_data["values"])
            parameters[name] = param
        
        return parameters
    
    def _adaptive_compression(self, parameters: Dict[str, jnp.ndarray]) -> Tuple[bytes, Dict[str, Any]]:
        """Adaptive compression - choose best method per parameter."""
        compressed_parts = {}
        compression_methods = {}
        
        for name, param in parameters.items():
            # Try different methods and choose best
            methods_to_try = [
                (CompressionType.GZIP, self._gzip_compression),
                (CompressionType.QUANTIZATION, self._quantization_compression),
                (CompressionType.SPARSIFICATION, self._sparsification_compression)
            ]
            
            best_size = float('inf')
            best_method = CompressionType.GZIP
            best_data = None
            best_meta = None
            
            single_param = {name: param}
            
            for method_type, method_func in methods_to_try:
                try:
                    data, meta = method_func(single_param)
                    if len(data) < best_size:
                        best_size = len(data)
                        best_method = method_type
                        best_data = data
                        best_meta = meta
                except Exception as e:
                    self.logger.warning(f"Compression method {method_type} failed for {name}: {e}")
            
            compressed_parts[name] = best_data
            compression_methods[name] = best_method.value
        
        # Combine all compressed parts
        combined_data = {
            "parts": compressed_parts,
            "methods": compression_methods
        }
        
        serialized = serialization.to_bytes(combined_data)
        final_compressed = zlib.compress(serialized)
        
        metadata = {
            "compression_type": CompressionType.ADAPTIVE.value,
            "methods_used": compression_methods,
            "original_size": sum(param.nbytes for param in parameters.values()),
            "compressed_size": len(final_compressed)
        }
        
        return final_compressed, metadata
    
    def get_compression_stats(self) -> Dict[str, float]:
        """Get compression statistics."""
        if not self.original_sizes or not self.compressed_sizes:
            return {"avg_compression_ratio": 1.0, "total_savings_mb": 0.0}
        
        avg_ratio = sum(
            comp / orig for orig, comp in zip(self.original_sizes, self.compressed_sizes)
        ) / len(self.original_sizes)
        
        total_savings = sum(self.original_sizes) - sum(self.compressed_sizes)
        
        return {
            "avg_compression_ratio": avg_ratio,
            "total_savings_mb": total_savings / (1024 * 1024),
            "total_original_mb": sum(self.original_sizes) / (1024 * 1024),
            "total_compressed_mb": sum(self.compressed_sizes) / (1024 * 1024)
        }


class EfficientAggregator:
    """
    Efficient parameter aggregation with various strategies.
    """
    
    def __init__(self, strategy: AggregationStrategy = AggregationStrategy.FEDAVG):
        self.strategy = strategy
        self.logger = get_logger("pg_neo_graph_rl.aggregator")
        
        # Strategy-specific parameters
        self.fedprox_mu = 0.01  # Proximal term weight
        self.scaffold_control_variates: Dict[str, jnp.ndarray] = {}
        self.trimmed_mean_beta = 0.1  # Fraction to trim
        
    def aggregate(self, 
                 parameter_updates: List[Dict[str, jnp.ndarray]],
                 weights: Optional[List[float]] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> Dict[str, jnp.ndarray]:
        """
        Aggregate parameter updates from multiple agents.
        
        Args:
            parameter_updates: List of parameter dictionaries
            weights: Optional weights for each update
            metadata: Additional metadata for aggregation
            
        Returns:
            Aggregated parameters
        """
        if not parameter_updates:
            raise ValueError("No parameter updates provided")
        
        if weights is None:
            weights = [1.0 / len(parameter_updates)] * len(parameter_updates)
        
        if len(weights) != len(parameter_updates):
            raise ValueError("Weights length must match parameter updates length")
        
        if self.strategy == AggregationStrategy.FEDAVG:
            return self._fedavg_aggregation(parameter_updates, weights)
        
        elif self.strategy == AggregationStrategy.FEDPROX:
            return self._fedprox_aggregation(parameter_updates, weights, metadata)
        
        elif self.strategy == AggregationStrategy.TRIMMED_MEAN:
            return self._trimmed_mean_aggregation(parameter_updates)
        
        elif self.strategy == AggregationStrategy.MEDIAN:
            return self._median_aggregation(parameter_updates)
        
        elif self.strategy == AggregationStrategy.KRUM:
            return self._krum_aggregation(parameter_updates)
        
        else:
            # Fallback to FedAvg
            return self._fedavg_aggregation(parameter_updates, weights)
    
    def _fedavg_aggregation(self, 
                           parameter_updates: List[Dict[str, jnp.ndarray]], 
                           weights: List[float]) -> Dict[str, jnp.ndarray]:
        """Standard FedAvg aggregation."""
        aggregated = {}
        
        for param_name in parameter_updates[0].keys():
            weighted_sum = jnp.zeros_like(parameter_updates[0][param_name])
            
            for update, weight in zip(parameter_updates, weights):
                weighted_sum += weight * update[param_name]
            
            aggregated[param_name] = weighted_sum
        
        return aggregated
    
    def _trimmed_mean_aggregation(self, 
                                 parameter_updates: List[Dict[str, jnp.ndarray]]) -> Dict[str, jnp.ndarray]:
        """Trimmed mean aggregation for Byzantine robustness."""
        aggregated = {}
        trim_count = int(len(parameter_updates) * self.trimmed_mean_beta)
        
        for param_name in parameter_updates[0].keys():
            # Stack all parameter values
            stacked_params = jnp.stack([update[param_name] for update in parameter_updates])
            
            # Sort along agent dimension and trim extremes
            sorted_params = jnp.sort(stacked_params, axis=0)
            trimmed_params = sorted_params[trim_count:-trim_count] if trim_count > 0 else sorted_params
            
            # Take mean of remaining values
            aggregated[param_name] = jnp.mean(trimmed_params, axis=0)
        
        return aggregated
    
    def _median_aggregation(self, 
                           parameter_updates: List[Dict[str, jnp.ndarray]]) -> Dict[str, jnp.ndarray]:
        """Coordinate-wise median aggregation."""
        aggregated = {}
        
        for param_name in parameter_updates[0].keys():
            stacked_params = jnp.stack([update[param_name] for update in parameter_updates])
            aggregated[param_name] = jnp.median(stacked_params, axis=0)
        
        return aggregated
    
    def _krum_aggregation(self, 
                         parameter_updates: List[Dict[str, jnp.ndarray]]) -> Dict[str, jnp.ndarray]:
        """Krum aggregation for Byzantine robustness."""
        if len(parameter_updates) < 3:
            # Fall back to FedAvg for small numbers
            return self._fedavg_aggregation(parameter_updates, [1.0/len(parameter_updates)] * len(parameter_updates))
        
        # Calculate pairwise distances
        distances = []
        for i, update_i in enumerate(parameter_updates):
            total_distance = 0
            for j, update_j in enumerate(parameter_updates):
                if i != j:
                    param_distance = 0
                    for param_name in update_i.keys():
                        param_distance += jnp.sum((update_i[param_name] - update_j[param_name]) ** 2)
                    total_distance += param_distance
            distances.append(total_distance)
        
        # Select update with smallest total distance to others
        krum_index = jnp.argmin(jnp.array(distances))
        return parameter_updates[krum_index]


class OptimizedCommunicationProtocol:
    """
    Optimized communication protocol for federated learning.
    """
    
    def __init__(self, 
                 compression_config: CompressionConfig,
                 aggregation_strategy: AggregationStrategy = AggregationStrategy.FEDAVG,
                 batch_size: int = 10,
                 enable_pipelining: bool = True):
        self.compression_config = compression_config
        self.aggregation_strategy = aggregation_strategy
        self.batch_size = batch_size
        self.enable_pipelining = enable_pipelining
        
        # Components
        self.compressor = ParameterCompressor(compression_config)
        self.aggregator = EfficientAggregator(aggregation_strategy)
        
        # Communication state
        self.pending_updates: Dict[str, Dict[str, Any]] = {}
        self.communication_metrics = CommunicationMetrics()
        self.lock = threading.RLock()
        
        # Bandwidth optimization
        self.adaptive_batch_size = batch_size
        self.latency_history: deque = deque(maxlen=100)
        
        self.logger = get_logger("pg_neo_graph_rl.comm_protocol")
    
    @retry_with_backoff(RetryConfig(max_attempts=3, base_delay=1.0))
    async def send_parameters(self, 
                            agent_id: str,
                            parameters: Dict[str, jnp.ndarray],
                            destination: str) -> bool:
        """
        Send parameters with compression and error handling.
        
        Args:
            agent_id: Sending agent ID
            parameters: Parameters to send
            destination: Destination identifier
            
        Returns:
            True if sent successfully
        """
        start_time = time.time()
        
        try:
            # Compress parameters
            compressed_data, metadata = self.compressor.compress(parameters)
            
            # Create message
            message = {
                "agent_id": agent_id,
                "compressed_data": compressed_data,
                "metadata": metadata,
                "timestamp": start_time
            }
            
            # Simulate network transmission (in real implementation, this would be actual network code)
            await self._simulate_network_send(message, destination)
            
            # Update metrics
            with self.lock:
                self.communication_metrics.bytes_sent += len(compressed_data)
                self.communication_metrics.compression_ratio = (
                    metadata.get("compressed_size", len(compressed_data)) / 
                    metadata.get("original_size", len(compressed_data))
                )
            
            latency = (time.time() - start_time) * 1000  # Convert to ms
            self.latency_history.append(latency)
            
            self.logger.debug(f"Sent parameters from {agent_id} to {destination} "
                            f"({len(compressed_data)} bytes, {latency:.1f}ms)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send parameters from {agent_id}: {e}")
            with self.lock:
                self.communication_metrics.error_count += 1
            raise CommunicationError(f"Parameter transmission failed: {e}", sender_id=agent_id)
    
    async def receive_parameters(self, 
                               message: Dict[str, Any]) -> Tuple[str, Dict[str, jnp.ndarray]]:
        """
        Receive and decompress parameters.
        
        Args:
            message: Received message
            
        Returns:
            Tuple of (agent_id, parameters)
        """
        try:
            agent_id = message["agent_id"]
            compressed_data = message["compressed_data"]
            metadata = message["metadata"]
            
            # Decompress parameters
            parameters = self.compressor.decompress(compressed_data, metadata)
            
            # Update metrics
            with self.lock:
                self.communication_metrics.bytes_received += len(compressed_data)
            
            self.logger.debug(f"Received parameters from {agent_id} "
                            f"({len(compressed_data)} bytes)")
            
            return agent_id, parameters
            
        except Exception as e:
            self.logger.error(f"Failed to receive parameters: {e}")
            raise CommunicationError(f"Parameter reception failed: {e}")
    
    async def aggregate_round(self, 
                            agent_updates: Dict[str, Dict[str, jnp.ndarray]],
                            weights: Optional[Dict[str, float]] = None) -> Dict[str, jnp.ndarray]:
        """
        Perform aggregation round with multiple agent updates.
        
        Args:
            agent_updates: Dictionary mapping agent IDs to parameter updates
            weights: Optional weights for each agent
            
        Returns:
            Aggregated parameters
        """
        if not agent_updates:
            raise ValueError("No agent updates provided")
        
        parameter_list = list(agent_updates.values())
        weight_list = None
        
        if weights:
            weight_list = [weights.get(agent_id, 1.0) for agent_id in agent_updates.keys()]
            # Normalize weights
            total_weight = sum(weight_list)
            weight_list = [w / total_weight for w in weight_list]
        
        # Perform aggregation
        aggregated = self.aggregator.aggregate(parameter_list, weight_list)
        
        self.logger.info(f"Aggregated parameters from {len(agent_updates)} agents")
        return aggregated
    
    async def batched_communication_round(self, 
                                        agent_updates: Dict[str, Dict[str, jnp.ndarray]],
                                        coordinator_id: str) -> Dict[str, jnp.ndarray]:
        """
        Perform batched communication round for efficiency.
        
        Args:
            agent_updates: Agent parameter updates
            coordinator_id: Coordinator agent ID
            
        Returns:
            Aggregated parameters
        """
        start_time = time.time()
        
        # Adaptive batch sizing based on network conditions
        current_batch_size = self._calculate_adaptive_batch_size()
        
        # Split agents into batches
        agent_ids = list(agent_updates.keys())
        batches = [
            agent_ids[i:i + current_batch_size] 
            for i in range(0, len(agent_ids), current_batch_size)
        ]
        
        # Process batches
        all_updates = {}
        
        for batch_idx, batch_agents in enumerate(batches):
            batch_updates = {
                agent_id: agent_updates[agent_id] 
                for agent_id in batch_agents
            }
            
            # Send batch to coordinator
            if self.enable_pipelining and batch_idx > 0:
                # Pipeline: start next batch while processing current
                await asyncio.gather(
                    self._process_batch(batch_updates, coordinator_id),
                    self._prepare_next_batch(batches[batch_idx + 1] if batch_idx + 1 < len(batches) else [])
                )
            else:
                await self._process_batch(batch_updates, coordinator_id)
            
            all_updates.update(batch_updates)
        
        # Aggregate all updates
        aggregated = await self.aggregate_round(all_updates)
        
        total_time = time.time() - start_time
        self.logger.info(f"Batched communication round completed in {total_time:.2f}s "
                        f"({len(batches)} batches, {len(agent_updates)} agents)")
        
        return aggregated
    
    def _calculate_adaptive_batch_size(self) -> int:
        """Calculate adaptive batch size based on network conditions."""
        if not self.latency_history:
            return self.batch_size
        
        avg_latency = sum(self.latency_history) / len(self.latency_history)
        
        # Adjust batch size based on latency
        if avg_latency < 100:  # Low latency - can use larger batches
            self.adaptive_batch_size = min(self.batch_size * 2, 20)
        elif avg_latency > 1000:  # High latency - use smaller batches
            self.adaptive_batch_size = max(self.batch_size // 2, 2)
        else:
            self.adaptive_batch_size = self.batch_size
        
        return self.adaptive_batch_size
    
    async def _process_batch(self, batch_updates: Dict[str, Dict[str, jnp.ndarray]], coordinator_id: str):
        """Process a batch of updates."""
        # Simulate batch processing
        await asyncio.sleep(0.1)  # Simulated processing time
    
    async def _prepare_next_batch(self, next_batch_agents: List[str]):
        """Prepare next batch for pipelining."""
        # Simulate preparation work
        await asyncio.sleep(0.05)
    
    async def _simulate_network_send(self, message: Dict[str, Any], destination: str):
        """Simulate network transmission."""
        # Simulate network latency based on data size
        data_size = len(message["compressed_data"])
        base_latency = 0.01  # 10ms base latency
        size_latency = data_size / (10 * 1024 * 1024)  # 10 MB/s bandwidth
        
        total_latency = base_latency + size_latency
        await asyncio.sleep(total_latency)
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get comprehensive communication statistics."""
        with self.lock:
            avg_latency = sum(self.latency_history) / len(self.latency_history) if self.latency_history else 0
            
            compression_stats = self.compressor.get_compression_stats()
            
            return {
                "bytes_sent": self.communication_metrics.bytes_sent,
                "bytes_received": self.communication_metrics.bytes_received,
                "total_bytes": self.communication_metrics.bytes_sent + self.communication_metrics.bytes_received,
                "compression_ratio": self.communication_metrics.compression_ratio,
                "avg_latency_ms": avg_latency,
                "error_count": self.communication_metrics.error_count,
                "adaptive_batch_size": self.adaptive_batch_size,
                "compression_stats": compression_stats,
                "aggregation_strategy": self.aggregation_strategy.value
            }


def create_optimized_protocol(compression_ratio: float = 0.1, 
                            aggregation_strategy: AggregationStrategy = AggregationStrategy.FEDAVG) -> OptimizedCommunicationProtocol:
    """
    Create optimized communication protocol with best practices.
    
    Args:
        compression_ratio: Target compression ratio
        aggregation_strategy: Aggregation strategy to use
        
    Returns:
        Configured communication protocol
    """
    compression_config = CompressionConfig(
        compression_type=CompressionType.ADAPTIVE,
        compression_ratio=compression_ratio,
        quantization_bits=8,
        sparsity_ratio=0.9
    )
    
    protocol = OptimizedCommunicationProtocol(
        compression_config=compression_config,
        aggregation_strategy=aggregation_strategy,
        batch_size=10,
        enable_pipelining=True
    )
    
    logger.info(f"Created optimized communication protocol with {aggregation_strategy.value} aggregation")
    return protocol