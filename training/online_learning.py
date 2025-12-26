"""
Online Learning Module with Elastic Weight Consolidation.

Enables continual learning in production by updating the model
on recent trading outcomes while preventing catastrophic forgetting
of previously learned patterns.

Reference: "Overcoming Catastrophic Forgetting in Neural Networks" 
(Kirkpatrick et al., 2017) https://www.pnas.org/doi/10.1073/pnas.1611835114

Key components:
- EWCLoss: Penalty term to preserve important parameters
- FisherInformation: Diagonal approximation of Fisher Information Matrix
- OnlineLearningModule: Orchestrates continual learning updates
- ReplayBuffer: Stores recent experiences for rehearsal

ARCHITECTURAL PRINCIPLE:
Online learning is SEPARATE from offline training. The production model
can receive small incremental updates based on recent trades without
requiring full retraining. EWC ensures these updates don't destroy
previously learned patterns.

Example:
    >>> from training.online_learning import OnlineLearningModule
    >>> online = OnlineLearningModule(model, ewc_lambda=0.4)
    >>> online.register_task()  # Snapshot current weights
    >>> online.update(recent_experiences)  # Incremental learning
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """
    Single trading experience for replay.
    
    Attributes:
        ticks: Tick features at decision time
        candles: Candle features at decision time
        vol_metrics: Volatility metrics
        contract_type: Type of contract traded
        probability: Model's predicted probability
        outcome: Actual outcome (1=win, 0=loss, -1=unknown)
        reconstruction_error: Regime detection error
        timestamp: Unix timestamp of trade
    """
    ticks: torch.Tensor
    candles: torch.Tensor
    vol_metrics: torch.Tensor
    contract_type: str
    probability: float
    outcome: int
    reconstruction_error: float
    timestamp: float


class ReplayBuffer:
    """
    Fixed-size buffer for storing recent trading experiences.
    
    Implements FIFO eviction when buffer is full.
    Provides sampling for mini-batch updates.
    """
    
    def __init__(self, capacity: int = 1000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum experiences to store
        """
        self.capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)
    
    def add(self, experience: Experience) -> None:
        """Add experience to buffer."""
        self._buffer.append(experience)
    
    def sample(self, batch_size: int) -> list[Experience]:
        """
        Sample random batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            List of sampled experiences
        """
        import random
        if len(self._buffer) < batch_size:
            return list(self._buffer)
        return random.sample(list(self._buffer), batch_size)
    
    def get_resolved_experiences(self) -> list[Experience]:
        """Get all experiences with known outcomes."""
        return [e for e in self._buffer if e.outcome >= 0]
    
    def __len__(self) -> int:
        return len(self._buffer)
    
    def clear(self) -> None:
        """Clear all experiences."""
        self._buffer.clear()


class FisherInformation:
    """
    Diagonal Fisher Information Matrix approximation.
    
    The Fisher Information measures the importance of each parameter
    for the current task. Parameters with high Fisher values are
    "important" and should be preserved during incremental updates.
    
    Uses diagonal approximation for efficiency (O(n) storage vs O(n²)).
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize Fisher Information tracker.
        
        Args:
            model: PyTorch model to track
        """
        self.model = model
        self._fisher: dict[str, torch.Tensor] = {}
        self._optimal_params: dict[str, torch.Tensor] = {}
    
    def compute(
        self,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        num_samples: int | None = None,
    ) -> None:
        """
        Compute diagonal Fisher Information.
        
        Uses the empirical Fisher which is the squared gradient of the loss.
        
        Args:
            dataloader: Data loader for experiences
            loss_fn: Loss function for computing gradients
            num_samples: Number of samples to use (None = all)
        """
        # Store optimal parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._optimal_params[name] = param.data.clone()
        
        # Initialize Fisher accumulator
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._fisher[name] = torch.zeros_like(param)
        
        self.model.train()
        
        n_samples = 0
        for batch in dataloader:
            if num_samples and n_samples >= num_samples:
                break
            
            # Forward pass
            inputs, targets = batch[:-1], batch[-1]
            outputs = self.model(*inputs)
            
            # Get loss based on output type
            if isinstance(outputs, dict):
                # Multi-head output
                loss = sum(
                    loss_fn(outputs[k].squeeze(), targets)
                    for k in outputs if "logit" in k
                )
            else:
                # Squeeze to match target shape
                if outputs.dim() > 1 and outputs.size(-1) == 1:
                    outputs = outputs.squeeze(-1)
                loss = loss_fn(outputs, targets)
            
            # Backward to get gradients
            self.model.zero_grad()
            loss.backward()
            
            # Accumulate squared gradients (Fisher diagonal)
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self._fisher[name] += param.grad.data ** 2
            
            n_samples += len(targets)
        
        # Normalize by number of samples
        for name in self._fisher:
            self._fisher[name] /= n_samples
        
        logger.info(f"Computed Fisher Information over {n_samples} samples")
    
    def get_fisher(self, name: str) -> torch.Tensor | None:
        """Get Fisher values for a parameter."""
        return self._fisher.get(name)
    
    def get_optimal_params(self, name: str) -> torch.Tensor | None:
        """Get optimal parameter values."""
        return self._optimal_params.get(name)
    
    def state_dict(self) -> dict[str, Any]:
        """Get state dictionary."""
        return {
            "fisher": self._fisher,
            "optimal_params": self._optimal_params,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dictionary."""
        self._fisher = state_dict["fisher"]
        self._optimal_params = state_dict["optimal_params"]
        logger.info(f"Loaded Fisher Information for {len(self._fisher)} parameters")

    def is_computed(self) -> bool:
        """Check if Fisher has been computed."""
        return len(self._fisher) > 0


class EWCLoss(nn.Module):
    """
    Elastic Weight Consolidation loss penalty.
    
    Adds a quadratic penalty for deviating from optimal parameters,
    weighted by the Fisher Information (importance) of each parameter:
    
    L_EWC = λ * Σᵢ Fᵢ * (θᵢ - θ*ᵢ)²
    
    Where:
    - λ = EWC strength parameter
    - Fᵢ = Fisher Information for parameter i
    - θᵢ = Current parameter value
    - θ*ᵢ = Optimal parameter value from previous task
    """
    
    def __init__(self, ewc_lambda: float = 0.4):
        """
        Initialize EWC loss.
        
        Args:
            ewc_lambda: Weight for EWC penalty (higher = more conservative)
        """
        super().__init__()
        self.ewc_lambda = ewc_lambda
    
    def forward(
        self,
        model: nn.Module,
        fisher_info: FisherInformation,
    ) -> torch.Tensor:
        """
        Compute EWC penalty.
        
        Args:
            model: Current model
            fisher_info: FisherInformation with stored Fisher and optimal params
        
        Returns:
            EWC penalty term
        """
        if not fisher_info.is_computed():
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        penalty = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher = fisher_info.get_fisher(name)
                optimal = fisher_info.get_optimal_params(name)
                
                if fisher is not None and optimal is not None:
                    penalty += (fisher * (param - optimal) ** 2).sum()
        
        return self.ewc_lambda * penalty


class OnlineLearningModule:
    """
    Orchestrates online (continual) learning for production models.
    
    Workflow:
    1. After successful training, call register_task() to snapshot state
    2. In production, add experiences to replay buffer
    3. Periodically call update() to incrementally improve on recent data
    
    The EWC constraint prevents catastrophic forgetting of original training.
    
    Example:
        >>> online = OnlineLearningModule(model, optimizer)
        >>> online.register_task(training_dataloader)  # After initial training
        >>> 
        >>> # In production loop:
        >>> online.add_experience(experience)
        >>> if online.should_update():
        ...     metrics = online.update()
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        ewc_lambda: float = 0.4,
        replay_capacity: int = 1000,
        update_interval: int = 100,
        min_experiences: int = 50,
        learning_rate: float = 1e-5,
    ):
        """
        Initialize online learning module.
        
        Args:
            model: Model to update
            optimizer: Optimizer (created if None)
            ewc_lambda: EWC penalty strength
            replay_capacity: Size of replay buffer
            update_interval: Experiences between updates
            min_experiences: Minimum resolved experiences for update
            learning_rate: Learning rate for online updates
        """
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.update_interval = update_interval
        self.min_experiences = min_experiences
        
        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=1e-5,
            )
        else:
            self.optimizer = optimizer
        
        # Components
        self.fisher = FisherInformation(model)
        self.ewc_loss = EWCLoss(ewc_lambda)
        self.replay_buffer = ReplayBuffer(replay_capacity)
        
        # State
        self._experiences_since_update = 0
        self._total_updates = 0
        
        logger.info(
            f"OnlineLearningModule initialized: "
            f"ewc_lambda={ewc_lambda}, capacity={replay_capacity}"
        )
    
    def register_task(
        self,
        dataloader: DataLoader | None = None,
        loss_fn: nn.Module | None = None,
    ) -> None:
        """
        Register current task and compute Fisher Information.
        
        Call this after initial training to establish the baseline
        that EWC will protect during online updates.
        
        Args:
            dataloader: Training data for Fisher computation
            loss_fn: Loss function for gradients
        """
        if dataloader is not None and loss_fn is not None:
            self.fisher.compute(dataloader, loss_fn)
        else:
            # Just snapshot parameters without Fisher
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.fisher._optimal_params[name] = param.data.clone()
        
        logger.info("Registered task for EWC protection")
    
    def add_experience(self, experience: Experience) -> None:
        """
        Add a trading experience to the replay buffer.
        
        Args:
            experience: Trading experience with outcome
        """
        self.replay_buffer.add(experience)
        self._experiences_since_update += 1
    
    def should_update(self) -> bool:
        """
        Check if it's time for an online update.
        
        Returns:
            True if update should be performed
        """
        resolved = len(self.replay_buffer.get_resolved_experiences())
        return (
            self._experiences_since_update >= self.update_interval
            and resolved >= self.min_experiences
        )
    
    def update(
        self,
        batch_size: int = 32,
        num_steps: int = 5,
        loss_fn: nn.Module | None = None,
    ) -> dict[str, float]:
        """
        Perform online update with EWC regularization.
        
        Args:
            batch_size: Batch size for update
            num_steps: Number of gradient steps
            loss_fn: Task loss function
        
        Returns:
            Dictionary of update metrics
        """
        if loss_fn is None:
            loss_fn = nn.BCEWithLogitsLoss()
        
        resolved = self.replay_buffer.get_resolved_experiences()
        if len(resolved) < self.min_experiences:
            logger.warning("Not enough resolved experiences for update")
            return {"skipped": True}
        
        self.model.train()
        
        total_loss = 0.0
        total_task_loss = 0.0
        total_ewc_loss = 0.0
        
        failed_steps = 0
        successful_steps = 0

        for step in range(num_steps):
            try:
                # Sample batch
                batch = self.replay_buffer.sample(min(batch_size, len(resolved)))
                
                # Prepare tensors
                ticks = torch.stack([e.ticks for e in batch])
                candles = torch.stack([e.candles for e in batch])
                vol_metrics = torch.stack([e.vol_metrics for e in batch])
                outcomes = torch.tensor([e.outcome for e in batch], dtype=torch.float32)
                
                # Forward pass
                outputs = self.model(ticks, candles, vol_metrics)
                
                # Task loss - use rise_fall_logit as primary
                if isinstance(outputs, dict) and "rise_fall_logit" in outputs:
                    predictions = outputs["rise_fall_logit"].squeeze()
                else:
                    predictions = outputs.squeeze() if hasattr(outputs, 'squeeze') else outputs
                
                task_loss = loss_fn(predictions, outcomes)
                
                # EWC penalty
                ewc_penalty = self.ewc_loss(self.model, self.fisher)
                
                # Total loss
                loss = task_loss + ewc_penalty
                
                # NaN/Inf Check
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Skipping step {step}: Loss is NaN/Inf")
                    failed_steps += 1
                    continue

                # Backward and update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_task_loss += task_loss.item()
                total_ewc_loss += ewc_penalty.item()
                successful_steps += 1
                
            except Exception as e:
                logger.error(f"Error in online update step {step}: {e}")
                failed_steps += 1
                self.optimizer.zero_grad()
        
        if successful_steps == 0:
            logger.error("All update steps failed")
            return {"skipped": True, "failed_steps": failed_steps}
        
        self._experiences_since_update = 0
        self._total_updates += 1
        
        divisor = successful_steps if successful_steps > 0 else 1
        metrics = {
            "total_loss": total_loss / divisor,
            "task_loss": total_task_loss / divisor,
            "ewc_loss": total_ewc_loss / divisor,
            "num_experiences": len(resolved),
            "update_count": self._total_updates,
        }
        
        logger.info(
            f"Online update #{self._total_updates}: "
            f"task_loss={metrics['task_loss']:.4f}, ewc_loss={metrics['ewc_loss']:.4f}"
        )
        
        return metrics
    
    def get_statistics(self) -> dict[str, Any]:
        """Get module statistics."""
        return {
            "buffer_size": len(self.replay_buffer),
            "resolved_experiences": len(self.replay_buffer.get_resolved_experiences()),
            "total_updates": self._total_updates,
            "experiences_since_update": self._experiences_since_update,
            "fisher_computed": self.fisher.is_computed(),
        }
