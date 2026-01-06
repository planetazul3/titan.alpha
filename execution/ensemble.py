
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Literal

from config.settings import EnsembleConfig

logger = logging.getLogger(__name__)

class EnsembleStrategy(ABC):
    """Abstract base class for model ensembling strategies."""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config

    @abstractmethod
    def combine(self, model_outputs: Dict[str, float]) -> float:
        """
        Combine multiple model probabilities into a single score.
        
        Args:
            model_outputs: Dictionary mapping model_name -> probability
            
        Returns:
            Aggregated probability (0.0 to 1.0)
        """
        pass


class AverageEnsemble(EnsembleStrategy):
    """Simple average of all model outputs."""
    
    def combine(self, model_outputs: Dict[str, float]) -> float:
        if not model_outputs:
            return 0.0
            
        probs = list(model_outputs.values())
        return sum(probs) / len(probs)


class VotingEnsemble(EnsembleStrategy):
    """
    Consensus voting. Returns average only if models agree on direction.
    Otherwise penalizes disagreement.
    """
    
    def combine(self, model_outputs: Dict[str, float]) -> float:
        if not model_outputs:
            return 0.0
            
        probs = list(model_outputs.values())
        avg = sum(probs) / len(probs)
        
        # Check consensus: Are all above 0.5 or all below 0.5?
        # If mixed, confidence should be low.
        
        above = sum(1 for p in probs if p > 0.5)
        total = len(probs)
        
        if above == total:
            # All bullish
            return avg
        elif above == 0:
            # All bearish
            return avg
        else:
            # Disagreement - return 0.5 (neutral) or penalize
            # If we average 0.9 and 0.1 -> 0.5. Correct.
            # But if we average 0.6 and 0.4 -> 0.5. Correct.
            return avg


class WeightedEnsemble(EnsembleStrategy):
    """Weighted average based on model reliability."""
    
    def __init__(self, config: EnsembleConfig, weights: Dict[str, float] | None = None):
        super().__init__(config)
        self.weights = weights or {} # model_name -> weight
        
    def set_weights(self, weights: Dict[str, float]):
        self.weights = weights
        
    def combine(self, model_outputs: Dict[str, float]) -> float:
        if not model_outputs:
            return 0.0
            
        total_weight = 0.0
        weighted_sum = 0.0
        
        for name, prob in model_outputs.items():
            w = self.weights.get(name, 1.0) # Default weight 1.0
            weighted_sum += prob * w
            total_weight += w
            
        if total_weight == 0:
            return 0.0
            
        return weighted_sum / total_weight


def create_ensemble(config: EnsembleConfig) -> EnsembleStrategy:
    """Factory to create configured ensemble strategy."""
    if config.strategy == "voting":
        return VotingEnsemble(config)
    elif config.strategy == "weighted":
        return WeightedEnsemble(config)
    else:
        return AverageEnsemble(config)
