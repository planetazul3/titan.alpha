"""
Reinforcement Learning Trainer for Trading Policy.

Implements Soft Actor-Critic (SAC) training loop for the
trading policy network.

Key components:
- RLTrainer: Main training orchestrator
- ReplayMemory: Experience replay buffer
- PolicyUpdater: SAC update logic

ARCHITECTURAL PRINCIPLE:
RL training is SEPARATE from supervised training. The RL policy
learns from trading outcomes, while the prediction model learns
from historical patterns. They can be trained independently.

Example:
    >>> from training.rl_trainer import RLTrainer
    >>> trainer = RLTrainer(actor, critic, temperature)
    >>> for experience in trading_loop:
    ...     trainer.add_experience(experience)
    ...     if trainer.should_update():
    ...         metrics = trainer.update()
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from models.policy import TemperatureParameter, TradingActor, TradingCritic

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

logger = logging.getLogger(__name__)


@dataclass
class Transition:
    """
    Single RL transition (s, a, r, s', done).
    
    Attributes:
        state: Current state tensor
        action: Action taken
        reward: Reward received
        next_state: Next state tensor
        done: Episode termination flag
    """
    state: torch.Tensor
    action: torch.Tensor
    reward: float
    next_state: torch.Tensor
    done: bool


class ReplayMemory:
    """
    Experience replay buffer for off-policy RL.
    
    Stores transitions and provides random sampling for training.
    """
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)
    
    def push(self, transition: Transition) -> None:
        """Add transition to buffer."""
        self._buffer.append(transition)
    
    def sample(self, batch_size: int) -> list[Transition]:
        """Sample random batch of transitions."""
        import random
        buffer_size = len(self._buffer)
        if buffer_size < batch_size:
            return list(self._buffer)
        # Sample indices instead of converting full buffer to list
        indices = random.sample(range(buffer_size), batch_size)
        return [self._buffer[i] for i in indices]
    
    def __len__(self) -> int:
        return len(self._buffer)
    
    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()


class RLTrainer:
    """
    Soft Actor-Critic trainer for trading policy.
    
    Implements the full SAC algorithm:
    1. Critic update (minimize TD error)
    2. Actor update (maximize expected Q-value)
    3. Temperature update (entropy tuning)
    4. Target network soft update
    
    Example:
        >>> trainer = RLTrainer(actor, critic, temp)
        >>> trainer.add_experience(state, action, reward, next_state, done)
        >>> if len(trainer.memory) >= 256:
        ...     metrics = trainer.update()
    """
    
    def __init__(
        self,
        actor: "TradingActor",
        critic: "TradingCritic",
        temperature: "TemperatureParameter",
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        memory_size: int = 100000,
        update_every: int = 1,
        target_update_interval: int = 1,
    ):
        """
        Initialize RL trainer.
        
        Args:
            actor: Policy network (TradingActor)
            critic: Q-network (TradingCritic)
            temperature: Temperature parameter
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            alpha_lr: Temperature learning rate
            gamma: Discount factor
            tau: Soft update coefficient
            batch_size: Training batch size
            memory_size: Replay buffer capacity
            update_every: Steps between updates
            target_update_interval: Steps between target updates
        """
        self.actor = actor
        self.critic = critic
        self.temperature = temperature
        
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every
        self.target_update_interval = target_update_interval
        
        # Optimizers
        self.actor_optimizer = Adam(actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(critic.parameters(), lr=critic_lr)
        self.alpha_optimizer = Adam([temperature.log_alpha], lr=alpha_lr)
        
        # Target critic network
        self.target_critic = self._create_target_critic(critic)
        
        # Replay memory
        self.memory = ReplayMemory(memory_size)
        
        # Tracking
        self._step_count = 0
        self._update_count = 0
        
        logger.info(
            f"RLTrainer initialized: gamma={gamma}, tau={tau}, batch={batch_size}"
        )
    
    def _create_target_critic(self, critic: nn.Module) -> nn.Module:
        """Create target critic as copy of critic."""
        import copy
        target = copy.deepcopy(critic)
        for param in target.parameters():
            param.requires_grad = False
        return target
    
    def add_experience(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        """Add experience to replay buffer."""
        transition = Transition(
            state=state.detach(),
            action=action.detach(),
            reward=reward,
            next_state=next_state.detach(),
            done=done,
        )
        self.memory.push(transition)
        self._step_count += 1
    
    def should_update(self) -> bool:
        """Check if update should be performed."""
        return (
            len(self.memory) >= self.batch_size
            and self._step_count % self.update_every == 0
        )
    
    def update(self) -> dict[str, float]:
        """
        Perform one SAC update step.
        
        Returns:
            Dictionary with training metrics
        """
        if len(self.memory) < self.batch_size:
            return {"skipped": True}
        
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.stack([t.state for t in batch])
        actions = torch.stack([t.action for t in batch])
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32).unsqueeze(1)
        next_states = torch.stack([t.next_state for t in batch])
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32).unsqueeze(1)
        
        # Update critic
        critic_loss = self._update_critic(states, actions, rewards, next_states, dones)
        
        # Update actor
        actor_loss, log_pi = self._update_actor(states)
        
        # Update temperature
        alpha_loss = self._update_temperature(log_pi)
        
        # Soft update target network
        self._update_count += 1
        if self._update_count % self.target_update_interval == 0:
            self._soft_update_target()
        
        return {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": self.temperature.alpha.item(),
            "update_count": self._update_count,
        }
    
    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> float:
        """Update critic network."""
        with torch.no_grad():
            # Get next actions and log probs
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            
            # Compute target Q-values
            target_q1, target_q2 = self.target_critic(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = target_q - self.temperature.alpha * next_log_probs
            
            # Bellman target
            target_value = rewards + self.gamma * (1 - dones) * target_q
        
        # Current Q-values
        current_q1, current_q2 = self.critic(states, actions)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)
        
        # Update
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def _update_actor(
        self,
        states: torch.Tensor,
    ) -> tuple[float, torch.Tensor]:
        """Update actor network."""
        # Sample new actions
        actions, log_probs, _ = self.actor.sample(states)
        
        # Compute Q-values for new actions
        q1, q2 = self.critic(states, actions)
        min_q = torch.min(q1, q2)
        
        # Actor loss (maximize Q - alpha * log_pi)
        actor_loss = (self.temperature.alpha * log_probs - min_q).mean()
        
        # Update
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        return actor_loss.item(), log_probs.detach()
    
    def _update_temperature(self, log_probs: torch.Tensor) -> float:
        """Update temperature parameter."""
        alpha_loss = self.temperature.compute_loss(log_probs)
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        return alpha_loss.item()
    
    def _soft_update_target(self) -> None:
        """Soft update target critic network."""
        for target_param, param in zip(
            self.target_critic.parameters(),
            self.critic.parameters(),
            strict=True,
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "temperature": self.temperature.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "step_count": self._step_count,
            "update_count": self._update_count,
        }, path)
        logger.info(f"Saved RL checkpoint to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        # SECURITY: weights_only=False is needed to load optimizer states and custom objects.
        # Only load checkpoints from trusted sources (e.g., your own training runs).
        checkpoint = torch.load(path)
        
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        self.temperature.load_state_dict(checkpoint["temperature"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        self._step_count = checkpoint["step_count"]
        self._update_count = checkpoint["update_count"]
        
        logger.info(f"Loaded RL checkpoint from {path}")
    
    def get_statistics(self) -> dict[str, Any]:
        """Get training statistics."""
        return {
            "memory_size": len(self.memory),
            "step_count": self._step_count,
            "update_count": self._update_count,
            "alpha": self.temperature.alpha.item(),
        }
