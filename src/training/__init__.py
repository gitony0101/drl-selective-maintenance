"""
Training Package for Milestone 5 Point-Estimate Double DQN.

Exports:
- DDQNTrainer: Training loop with validation, checkpoint selection, artifacts
"""

from .ddqn_trainer import DDQNTrainer, TrainerConfig, TrainingMetrics

__all__ = [
    "DDQNTrainer",
    "TrainerConfig",
    "TrainingMetrics",
]