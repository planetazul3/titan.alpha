"""
Meta-Ensemble System for Model Selection and Combination.

Implements an ensemble approach where multiple models are
tracked and combined based on their recent performance.

Key components:
- ModelRegistry: Tracks available models and their metadata
- PerformanceTracker: Monitors model performance over time
- MetaLearner: Learns optimal model weights based on context
- EnsemblePredictor: Combines model predictions dynamically

ARCHITECTURAL PRINCIPLE:
The ensemble provides robustness by automatically weighting
models based on recent performance. When a model degrades,
its weight decreases automatically. When a new model is
trained, it can be A/B tested before full promotion.

Example:
    >>> from models.ensemble import EnsemblePredictor, ModelRegistry
    >>> registry = ModelRegistry()
    >>> registry.register("v1", model_v1)
    >>> registry.register("v2", model_v2)
    >>> ensemble = EnsemblePredictor(registry)
    >>> probs = ensemble.predict(ticks, candles, vol_metrics)
"""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TradingModel(Protocol):
    """Protocol for trading models in ensemble."""
    
    def predict_probs(
        self,
        ticks: torch.Tensor,
        candles: torch.Tensor,
        vol_metrics: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict probabilities for contracts."""
        ...
    
    def get_volatility_anomaly_score(
        self,
        vol_metrics: torch.Tensor,
    ) -> torch.Tensor:
        """Get reconstruction error for regime detection."""
        ...


@dataclass
class ModelMetadata:
    """
    Metadata for a registered model.
    
    Attributes:
        model_id: Unique identifier
        version: Model version string
        registered_at: Registration timestamp
        checkpoint_path: Path to model checkpoint
        training_metrics: Metrics from training
        is_primary: Whether this is the primary model
        weight: Current ensemble weight
    """
    model_id: str
    version: str
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    checkpoint_path: Path | None = None
    training_metrics: dict[str, float] = field(default_factory=dict)
    is_primary: bool = False
    weight: float = 1.0
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "registered_at": self.registered_at.isoformat(),
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "training_metrics": self.training_metrics,
            "is_primary": self.is_primary,
            "weight": self.weight,
        }


class PerformanceTracker:
    """
    Track model performance over time using rolling metrics.
    
    Monitors:
    - Accuracy (correct predictions)
    - Calibration (predicted prob matches outcome rate)
    - Profit factor (gains / losses)
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._predictions: deque = deque(maxlen=window_size)
        self._outcomes: deque = deque(maxlen=window_size)
        self._pnls: deque = deque(maxlen=window_size)
    
    def record(
        self,
        prediction: float,
        outcome: int,
        pnl: float | None = None,
    ) -> None:
        """Record a prediction with outcome."""
        self._predictions.append(prediction)
        self._outcomes.append(outcome)
        if pnl is not None:
            self._pnls.append(pnl)
    
    def get_accuracy(self, threshold: float = 0.5) -> float:
        """Get rolling accuracy."""
        if not self._predictions:
            return 0.5
        
        correct = sum(
            1 for p, o in zip(self._predictions, self._outcomes, strict=True)
            if (p >= threshold) == (o == 1)
        )
        return correct / len(self._predictions)
    
    def get_profit_factor(self) -> float:
        """Get profit factor (gains / losses)."""
        if not self._pnls:
            return 1.0
        
        gains = sum(p for p in self._pnls if p > 0)
        losses = abs(sum(p for p in self._pnls if p < 0))
        
        if losses == 0:
            return float("inf") if gains > 0 else 1.0
        return float(gains / losses)
    
    def get_win_rate(self) -> float:
        """Get win rate."""
        if not self._outcomes:
            return 0.5
        return cast(float, sum(self._outcomes) / len(self._outcomes))
    
    def get_score(self) -> float:
        """
        Get composite performance score.
        
        Combines multiple metrics into single score for ranking.
        """
        accuracy = self.get_accuracy()
        win_rate = self.get_win_rate()
        
        # Simple weighted combination
        return 0.6 * accuracy + 0.4 * win_rate


class ModelRegistry:
    """
    Registry for managing multiple trading models.
    
    Provides:
    - Model registration and lookup
    - Primary model designation
    - Metadata management
    - Performance tracking per model
    """
    
    def __init__(self):
        self._models: dict[str, nn.Module] = {}
        self._metadata: dict[str, ModelMetadata] = {}
        self._trackers: dict[str, PerformanceTracker] = {}
        self._primary_id: str | None = None
    
    def register(
        self,
        model_id: str,
        model: nn.Module,
        version: str = "1.0.0",
        checkpoint_path: Path | None = None,
        training_metrics: dict[str, float] | None = None,
        make_primary: bool = False,
    ) -> None:
        """
        Register a model in the registry.
        
        Args:
            model_id: Unique identifier for model
            model: PyTorch model instance
            version: Version string
            checkpoint_path: Path to checkpoint
            training_metrics: Training metrics
            make_primary: Set as primary model
        """
        self._models[model_id] = model
        self._metadata[model_id] = ModelMetadata(
            model_id=model_id,
            version=version,
            checkpoint_path=checkpoint_path,
            training_metrics=training_metrics or {},
            is_primary=make_primary,
        )
        self._trackers[model_id] = PerformanceTracker()
        
        if make_primary or self._primary_id is None:
            self._primary_id = model_id
            for mid, meta in self._metadata.items():
                meta.is_primary = (mid == model_id)
        
        logger.info(f"Registered model '{model_id}' (v{version}), primary={make_primary}")
    
    def get(self, model_id: str) -> nn.Module | None:
        """Get model by ID."""
        return self._models.get(model_id)
    
    def get_primary(self) -> nn.Module | None:
        """Get the primary model."""
        if self._primary_id:
            return self._models.get(self._primary_id)
        return None
    
    def get_metadata(self, model_id: str) -> ModelMetadata | None:
        """Get model metadata."""
        return self._metadata.get(model_id)
    
    def get_tracker(self, model_id: str) -> PerformanceTracker | None:
        """Get performance tracker for model."""
        return self._trackers.get(model_id)
    
    def list_models(self) -> list[str]:
        """List all registered model IDs."""
        return list(self._models.keys())
    
    def set_primary(self, model_id: str) -> None:
        """Set a model as primary."""
        if model_id not in self._models:
            raise ValueError(f"Model '{model_id}' not registered")
        
        self._primary_id = model_id
        for mid, meta in self._metadata.items():
            meta.is_primary = (mid == model_id)
    
    def remove(self, model_id: str) -> None:
        """Remove a model from registry."""
        if model_id == self._primary_id:
            raise ValueError("Cannot remove primary model")
        
        self._models.pop(model_id, None)
        self._metadata.pop(model_id, None)
        self._trackers.pop(model_id, None)
    
    def record_prediction(
        self,
        model_id: str,
        prediction: float,
        outcome: int,
        pnl: float | None = None,
    ) -> None:
        """Record prediction outcome for a model."""
        tracker = self._trackers.get(model_id)
        if tracker:
            tracker.record(prediction, outcome, pnl)


class MetaLearner(nn.Module):
    """
    Meta-learner for dynamic model weighting.
    
    Takes market context as input and outputs model weights
    for optimal combination.
    
    Architecture: Simple MLP that learns which model to prefer
    in different market conditions.
    """
    
    def __init__(
        self,
        context_dim: int,
        num_models: int,
        hidden_dim: int = 32,
    ):
        super().__init__()
        
        self.context_dim = context_dim
        self.num_models = num_models
        
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_models),
            nn.Softmax(dim=-1),
        )
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Compute model weights given context.
        
        Args:
            context: Market context [batch, context_dim]
        
        Returns:
            Model weights [batch, num_models] summing to 1
        """
        return cast(torch.Tensor, self.net(context))


class EnsemblePredictor:
    """
    Combines predictions from multiple models.
    
    Weighting strategies:
    - equal: All models weighted equally
    - performance: Weighted by recent performance
    - meta: Learned weights based on context
    
    Example:
        >>> ensemble = EnsemblePredictor(registry, strategy="performance")
        >>> probs = ensemble.predict(ticks, candles, vol)
    """
    
    def __init__(
        self,
        registry: ModelRegistry,
        strategy: str = "performance",
        meta_learner: MetaLearner | None = None,
    ):
        """
        Initialize ensemble predictor.
        
        Args:
            registry: Model registry
            strategy: Weighting strategy (equal, performance, meta)
            meta_learner: MetaLearner for meta strategy
        """
        self.registry = registry
        self.strategy = strategy
        self.meta_learner = meta_learner
        
        logger.info(f"EnsemblePredictor initialized with strategy={strategy}")
    
    def get_weights(
        self,
        context: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """
        Get current model weights.
        
        Args:
            context: Optional context for meta strategy
        
        Returns:
            Dictionary mapping model_id to weight
        """
        model_ids = self.registry.list_models()
        
        if not model_ids:
            return {}
        
        if self.strategy == "equal":
            weight = 1.0 / len(model_ids)
            return {mid: weight for mid in model_ids}
        
        elif self.strategy == "performance":
            scores = {}
            for mid in model_ids:
                tracker = self.registry.get_tracker(mid)
                scores[mid] = tracker.get_score() if tracker else 0.5
            
            # Softmax-like normalization
            total = sum(scores.values())
            if total > 0:
                return {mid: s / total for mid, s in scores.items()}
            return {mid: 1.0 / len(model_ids) for mid in model_ids}
        
        elif self.strategy == "meta" and self.meta_learner and context is not None:
            with torch.no_grad():
                weights = self.meta_learner(context.unsqueeze(0)).squeeze(0)
            return {mid: weights[i].item() for i, mid in enumerate(model_ids)}
        
        # Fallback to equal weights
        return {mid: 1.0 / len(model_ids) for mid in model_ids}
    
    def predict(
        self,
        ticks: torch.Tensor,
        candles: torch.Tensor,
        vol_metrics: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Get ensemble prediction.
        
        Args:
            ticks: Tick data
            candles: Candle data
            vol_metrics: Volatility metrics
            context: Optional context for meta weighting
        
        Returns:
            Weighted average probabilities
        """
        weights = self.get_weights(context)
        
        if not weights:
            raise ValueError("No models in registry")
        
        # Collect predictions from all models
        all_probs: dict[str, list] = defaultdict(list)
        model_weights = []
        
        for model_id, weight in weights.items():
            model = self.registry.get(model_id)
            if model is None:
                continue
            
            # Cast to protocol
            from typing import cast
            model_typed = cast(TradingModel, model)
            
            with torch.no_grad():
                probs = model_typed.predict_probs(ticks, candles, vol_metrics)
            
            for key, prob in probs.items():
                all_probs[key].append((weight, prob))
            model_weights.append(weight)
        
        # Weighted combination
        combined = {}
        for key, weighted_probs in all_probs.items():
            total_weight = sum(w for w, _ in weighted_probs)
            if total_weight > 0:
                combined[key] = sum(w * p for w, p in weighted_probs) / total_weight
            else:
                combined[key] = weighted_probs[0][1]  # Fallback to first
        
        return combined
    
    def predict_best(
        self,
        ticks: torch.Tensor,
        candles: torch.Tensor,
        vol_metrics: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], str]:
        """
        Get prediction from best-performing model.
        
        Returns:
            Tuple of (predictions, best_model_id)
        """
        model_ids = self.registry.list_models()
        
        if not model_ids:
            raise ValueError("No models in registry")
        
        # Find best model by performance
        best_id = model_ids[0]
        best_score = 0.0
        
        for mid in model_ids:
            tracker = self.registry.get_tracker(mid)
            if tracker:
                score = tracker.get_score()
                if score > best_score:
                    best_score = score
                    best_id = mid
        
        model = self.registry.get(best_id)
        if model is None:
             raise ValueError(f"Model {best_id} not found")

        # Cast to TradingModel protocol or assume it complies
        # For strict mypy: cast(TradingModel, model)
        # But for now, just assertion is enough for runtime safety, type checker needs cast
        from typing import cast
        model_typed = cast(TradingModel, model)
        
        with torch.no_grad():
            probs = model_typed.predict_probs(ticks, candles, vol_metrics)
        
        return probs, best_id
    
    def get_statistics(self) -> dict[str, Any]:
        """Get ensemble statistics."""
        stats: dict[str, Any] = {
            "strategy": self.strategy,
            "num_models": len(self.registry.list_models()),
            "models": {},
        }
        
        for mid in self.registry.list_models():
            tracker = self.registry.get_tracker(mid)
            metadata = self.registry.get_metadata(mid)
            stats["models"][mid] = {
                "version": metadata.version if metadata else "unknown",
                "is_primary": metadata.is_primary if metadata else False,
                "accuracy": tracker.get_accuracy() if tracker else 0.0,
                "win_rate": tracker.get_win_rate() if tracker else 0.0,
                "score": tracker.get_score() if tracker else 0.0,
            }
        
        return stats
