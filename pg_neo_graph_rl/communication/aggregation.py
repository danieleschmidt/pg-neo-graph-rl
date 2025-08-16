"""
Federated aggregation methods.
"""
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp


class FederatedAggregator:
    """Handles different federated aggregation strategies."""

    def __init__(self, aggregation_method: str = "fedavg"):
        self.aggregation_method = aggregation_method

    def aggregate(self,
                 client_params: List[Dict[str, jnp.ndarray]],
                 client_weights: Optional[List[float]] = None) -> Dict[str, jnp.ndarray]:
        """Aggregate parameters from multiple clients."""

        if not client_params:
            return {}

        if client_weights is None:
            client_weights = [1.0 / len(client_params)] * len(client_params)

        if self.aggregation_method == "fedavg":
            return self._federated_averaging(client_params, client_weights)
        elif self.aggregation_method == "median":
            return self._federated_median(client_params)
        elif self.aggregation_method == "trimmed_mean":
            return self._trimmed_mean(client_params, trim_ratio=0.1)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    def _federated_averaging(self,
                           client_params: List[Dict[str, jnp.ndarray]],
                           weights: List[float]) -> Dict[str, jnp.ndarray]:
        """Standard federated averaging (FedAvg)."""
        aggregated = {}

        for key in client_params[0].keys():
            # Weighted average of parameters
            weighted_params = [w * params[key] for w, params in zip(weights, client_params)]
            aggregated[key] = sum(weighted_params)

        return aggregated

    def _federated_median(self, client_params: List[Dict[str, jnp.ndarray]]) -> Dict[str, jnp.ndarray]:
        """Coordinate-wise median aggregation (robust to outliers)."""
        aggregated = {}

        for key in client_params[0].keys():
            param_stack = jnp.stack([params[key] for params in client_params])
            aggregated[key] = jnp.median(param_stack, axis=0)

        return aggregated

    def _trimmed_mean(self,
                     client_params: List[Dict[str, jnp.ndarray]],
                     trim_ratio: float = 0.1) -> Dict[str, jnp.ndarray]:
        """Trimmed mean aggregation (removes extreme values)."""
        aggregated = {}
        num_clients = len(client_params)
        trim_count = int(num_clients * trim_ratio)

        for key in client_params[0].keys():
            param_stack = jnp.stack([params[key] for params in client_params])

            # Sort and trim extreme values
            sorted_params = jnp.sort(param_stack, axis=0)
            trimmed_params = sorted_params[trim_count:num_clients-trim_count]

            aggregated[key] = jnp.mean(trimmed_params, axis=0)

        return aggregated

    def secure_aggregation(self,
                          client_params: List[Dict[str, jnp.ndarray]],
                          noise_scale: float = 0.01) -> Dict[str, jnp.ndarray]:
        """Secure aggregation with differential privacy."""
        # Add noise for privacy
        noisy_params = []
        for params in client_params:
            noisy_param = {}
            for key, param in params.items():
                noise = jax.random.normal(jax.random.PRNGKey(42), param.shape) * noise_scale
                noisy_param[key] = param + noise
            noisy_params.append(noisy_param)

        # Aggregate noisy parameters
        return self._federated_averaging(noisy_params, None)

    def adaptive_aggregation(self,
                           client_params: List[Dict[str, jnp.ndarray]],
                           client_losses: List[float]) -> Dict[str, jnp.ndarray]:
        """Adaptive aggregation based on client performance."""
        # Weight clients inversely by their loss (better clients get higher weight)
        min_loss = min(client_losses)
        weights = [(min_loss + 1e-8) / (loss + 1e-8) for loss in client_losses]

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        return self._federated_averaging(client_params, weights)
