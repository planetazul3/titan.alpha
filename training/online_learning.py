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
    
    def sample(self, batch_size: int, stratified: bool = True) -> list[Experience]:
        """
        Sample random batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            stratified: If True, stratifies by volatility (Low/Med/High)
        
        Returns:
            List of sampled experiences
        """
        import random
        if len(self._buffer) < batch_size:
            return list(self._buffer)
            
        if not stratified:
            return random.sample(list(self._buffer), batch_size)
            
        # Stratified Sampling by Volatility Regime
        # We use a heuristic: 'vol_metrics' usually contains [vol_short, vol_long, ...]
        # We'll use the first metric (vol_short)
        
        low_vol = []
        med_vol = []
        high_vol = []
        
        # Calculate approximate quantiles from buffer (simplified: fixed thresholds or dynamic?)
        # For speed, let's use fixed thresholds derived from normalization assumption (z-score approx 0)
        # But data might not be z-scored here in raw form.
        # Let's use dynamic quantiles of the current buffer.
        
        all_vols = [e.vol_metrics[0].item() for e in self._buffer]
        if not all_vols:
             return random.sample(list(self._buffer), batch_size)

        # Efficient approximate partitioning
        # Sort is O(N log N), acceptable for buffer size ~1000-5000
        # For larger buffers, reservoir sampling per bucket is better.
        
        import statistics
        try:
             # Fast 33/66 percentiles
             # We can just sort and split
             sorted_indices = sorted(range(len(all_vols)), key=lambda k: all_vols[k])
             n = len(sorted_indices)
             t1 = int(n * 0.33)
             t2 = int(n * 0.66)
             
             # Map back to experiences
             # This is a bit heavy for every sample call if calc_quantiles every time.
             # Optimization: bucket on insert? 
             # For now, let's just do random sampling if not enough diversity, 
             # or simple bucket rejection?
             
             # Actually, simpler robust approach:
             # Just split into 3 groups based on min/max range of current buffer
             min_v = min(all_vols)
             max_v = max(all_vols)
             range_v = max_v - min_v
             if range_v < 1e-6:
                 return random.sample(list(self._buffer), batch_size)
                 
             b1_thresh = min_v + range_v * 0.33
             b2_thresh = min_v + range_v * 0.66
             
             buffer_list = list(self._buffer)
             for e in buffer_list:
                 v = e.vol_metrics[0].item()
                 if v <= b1_thresh:
                     low_vol.append(e)
                 elif v <= b2_thresh:
                     med_vol.append(e)
                 else:
                     high_vol.append(e)
        except Exception:
             # Fallback
             return random.sample(list(self._buffer), batch_size)
        
        # Stratified draw: try to get equal parts
        n_per_stratum = batch_size // 3
        remainder = batch_size % 3
        
        samples = []
        
        # Helper to safely sample
        def safe_sample(pool, k):
            if not pool: return []
            return random.sample(pool, min(len(pool), k))
            
        samples.extend(safe_sample(low_vol, n_per_stratum + (1 if remainder > 0 else 0)))
        samples.extend(safe_sample(med_vol, n_per_stratum + (1 if remainder > 1 else 0)))
        samples.extend(safe_sample(high_vol, n_per_stratum))
        
        # If we didn't fill the batch (due to empty buckets), fill from remainder
        if len(samples) < batch_size:
             needed = batch_size - len(samples)
             
             if len(self._buffer) > len(samples):
                 # Sample random indices not already chosen
                 chosen_ids = {id(e) for e in samples}
                 remaining_candidates = [e for e in self._buffer if id(e) not in chosen_ids]
                 
                 if remaining_candidates:
                    samples.extend(random.sample(remaining_candidates, min(len(remaining_candidates), needed)))
        
        return samples[:batch_size]
    
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
        alpha: float = 1.0,
    ) -> None:
        """
        Compute diagonal Fisher Information.
        
        Uses the empirical Fisher which is the squared gradient of the loss.
        
        Args:
            dataloader: Data loader for experiences
            loss_fn: Loss function for computing gradients
            num_samples: Number of samples to use (None = all)
            alpha: Decay factor for accumulation (1.0 = overwrite, <1.0 = moving average)
                   new_fisher = alpha * computed_fisher + (1 - alpha) * old_fisher
        """
        # Store optimal parameters (Snapshot current)
        for name, param in self.model.named_parameters():
             if param.requires_grad:
                 self._optimal_params[name] = param.data.clone()
        
        # Initialize Fisher accumulator for this RUN
        current_run_fisher = {}
        for name, param in self.model.named_parameters():
             if param.requires_grad:
                 current_run_fisher[name] = torch.zeros_like(param)
        
        self.model.train()
        
        n_samples = 0
        for batch in dataloader:
            if num_samples and n_samples >= num_samples:
                break
            
            # Forward pass - Handle both dict and tensor-list batches
            if isinstance(batch, dict):
                ticks = batch["ticks"].to(next(self.model.parameters()).device)
                candles = batch["candles"].to(next(self.model.parameters()).device)
                vol_metrics = batch["vol_metrics"].to(next(self.model.parameters()).device)
                targets = {k: v.to(next(self.model.parameters()).device) for k, v in batch["targets"].items()}
                
                outputs = self.model(ticks, candles, vol_metrics)
                
                # Multi-head loss for EWC
                # If using the standard multi-task criterion, pass dicts
                try:
                    loss_dict = loss_fn(outputs, targets, vol_metrics)
                    loss = loss_dict["total"] if isinstance(loss_dict, dict) else loss_dict
                except Exception:
                    # Fallback to summing Individual logit losses
                    loss = 0
                    for k in outputs:
                        if "logit" in k:
                            target_key = k.replace("_logit", "")
                            if target_key in targets:
                                loss += nn.functional.binary_cross_entropy_with_logits(
                                    outputs[k].squeeze(), targets[target_key].float()
                                )
            else:
                # Fallback for tensor datasets
                inputs, targets = batch[:-1], batch[-1]
                inputs = [i.to(next(self.model.parameters()).device) for i in inputs]
                targets = targets.to(next(self.model.parameters()).device)
                
                outputs = self.model(*inputs)
                
                if isinstance(outputs, dict):
                    loss = sum(
                        loss_fn(outputs[k].squeeze(), targets)
                        for k in outputs if "logit" in k
                    )
                else:
                    if outputs.dim() > 1 and outputs.size(-1) == 1:
                        outputs = outputs.squeeze(-1)
                    loss = loss_fn(outputs, targets)
            
            # Backward to get gradients
            self.model.zero_grad()
            loss.backward()
            
            # Accumulate squared gradients (Fisher diagonal)
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    current_run_fisher[name] += param.grad.data ** 2
            
            n_samples += len(targets)
        
        # Normalize by number of samples
        for name in current_run_fisher:
            current_run_fisher[name] /= n_samples
        
        # Clamp Fisher values to prevent extreme values that could destabilize EWC
        fisher_min = 1e-8  # Prevent division issues
        fisher_max = 1e6   # Prevent extreme penalties
        total_clamped = 0
        total_params = 0
        
        for name in current_run_fisher:
             original = current_run_fisher[name]
             clamped = torch.clamp(original, min=fisher_min, max=fisher_max)
             total_clamped += (original != clamped).sum().item()
             total_params += original.numel()
             
             # Apply Moving Average/Overwrite
             if name in self._fisher and alpha < 1.0:
                 self._fisher[name] = alpha * clamped + (1.0 - alpha) * self._fisher[name]
             else:
                 self._fisher[name] = clamped
        
        # Log Fisher statistics for debugging
        all_fisher = torch.cat([f.flatten() for f in self._fisher.values()])
        logger.info(
            f"Computed Fisher Information over {n_samples} samples. "
            f"Stats: mean={all_fisher.mean():.4e}, std={all_fisher.std():.4e}, "
            f"min={all_fisher.min():.4e}, max={all_fisher.max():.4e}, "
            f"clamped={total_clamped}/{total_params} ({100*total_clamped/total_params:.2f}%)"
        )
    
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
        update_interval_hours: float = 4.0,
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
            update_interval_hours: Time-based update trigger (hours)
        """
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.update_interval = update_interval
        self.min_experiences = min_experiences
        self.update_interval_hours = update_interval_hours
        
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
        self._last_update_time: float | None = None
        
        logger.info(
            f"OnlineLearningModule initialized: "
            f"ewc_lambda={ewc_lambda}, capacity={replay_capacity}, "
            f"update_interval_hours={update_interval_hours}"
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
        
        Uses a hybrid trigger: experience count OR time elapsed.
        
        Returns:
            True if update should be performed
        """
        import time
        
        resolved = len(self.replay_buffer.get_resolved_experiences())
        if resolved < self.min_experiences:
            return False
        
        # Experience-based trigger
        experience_trigger = self._experiences_since_update >= self.update_interval
        
        # Time-based trigger (R03)
        time_trigger = False
        if self._last_update_time is not None:
            elapsed_hours = (time.time() - self._last_update_time) / 3600.0
            time_trigger = elapsed_hours >= self.update_interval_hours
        else:
            # First update - use experience trigger only
            pass
        
        return experience_trigger or time_trigger
    
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
        
        import time
        
        self._experiences_since_update = 0
        self._total_updates += 1
        self._last_update_time = time.time()
        
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
