"""
Environment Step Semantics Audit for Milestone 5 DDQN.

Verifies:
1. trainer.global_step == agent.global_step after every step
2. One environment interaction increments counter exactly once
3. Gradient updates do NOT increment environment steps
4. Epsilon advances once per environment step
5. Epsilon is independent of warmup and update_frequency
6. gradient_update_count counts optimizer updates only
"""

import pytest

pytestmark = pytest.mark.requires_external_assets
import torch
import numpy as np
import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.ddqn_trainer import DDQNTrainer, TrainerConfig


class TestEnvironmentStepSemantics:
    """Audit environment-step semantics after agent.py changes."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)

    @pytest.fixture
    def bounded_trainer(self, temp_dir):
        """Create a bounded trainer for testing."""
        config = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            maintenance_capacity=1,
            cost_regime_id="failure-light-no-waste",
            max_steps=200,
            warmup_transitions=20,
            batch_size=16,
            update_frequency=2,  # Update every 2 steps
            validation_interval=500,
            checkpoint_interval=100,
            replay_capacity=1000,
            training_seed=6521,
            validation_seed=6521,
            output_dir=str(temp_dir / "test_run"),
        )
        return DDQNTrainer(config=config)

    def test_trainer_global_step_equals_agent_global_step(self, bounded_trainer):
        """Verify trainer.global_step == agent.global_step after every step."""
        obs, _ = bounded_trainer.train_env.reset()
        bounded_trainer._current_obs = obs

        for step in range(1, 51):
            bounded_trainer.train_step()

            # PROOF: trainer.global_step == agent.global_step
            assert bounded_trainer.global_step == bounded_trainer.agent.global_step, \
                f"Step {step}: trainer.global_step={bounded_trainer.global_step} != agent.global_step={bounded_trainer.agent.global_step}"
            assert bounded_trainer.global_step == step, \
                f"Step {step}: trainer.global_step={bounded_trainer.global_step} != expected {step}"

    def test_one_env_interaction_increments_once(self, bounded_trainer):
        """Verify one environment step increments counter exactly once."""
        obs, _ = bounded_trainer.train_env.reset()
        bounded_trainer._current_obs = obs

        initial_step = bounded_trainer.global_step

        # Single training step = single env interaction
        bounded_trainer.train_step()

        # PROOF: Exactly one increment
        assert bounded_trainer.global_step == initial_step + 1, \
            f"Expected exactly +1 increment, got {bounded_trainer.global_step - initial_step}"

    def test_gradient_updates_do_not_increment_env_steps(self, bounded_trainer):
        """Verify gradient updates do not increment environment step counter."""
        obs, _ = bounded_trainer.train_env.reset()
        bounded_trainer._current_obs = obs

        # Run enough steps to get gradient updates
        steps_before = 100
        for _ in range(steps_before):
            bounded_trainer.train_step()

        steps_after_grads = bounded_trainer.global_step
        grad_updates = bounded_trainer.agent.gradient_update_count

        # PROOF: global_step counts env steps, not gradient updates
        assert steps_after_grads == steps_before, \
            f"global_step should equal env steps ({steps_before}), got {steps_after_grads}"

        # gradient_update_count should be less (due to warmup and update_frequency)
        # With warmup=20 and update_frequency=2, we expect roughly (100-20)/2 = 40 updates
        assert bounded_trainer.agent.gradient_update_count < steps_after_grads, \
            f"gradient_update_count ({grad_updates}) should be < global_step ({steps_after_grads})"

    def test_epsilon_advances_per_env_step(self, bounded_trainer):
        """Verify epsilon advances once per environment step."""
        obs, _ = bounded_trainer.train_env.reset()
        bounded_trainer._current_obs = obs

        epsilon_before = bounded_trainer.agent.epsilon_state.epsilon
        epsilon_step_before = bounded_trainer.agent.epsilon_state.global_step

        # One environment step
        bounded_trainer.train_step()

        epsilon_after = bounded_trainer.agent.epsilon_state.epsilon
        epsilon_step_after = bounded_trainer.agent.epsilon_state.global_step

        # PROOF: epsilon advanced
        assert epsilon_step_after == epsilon_step_before + 1, \
            f"epsilon_state.global_step should advance by 1: {epsilon_step_before} -> {epsilon_step_after}"

        # Epsilon should decay (or stay at epsilon_end if already there)
        assert epsilon_after <= epsilon_before, \
            f"epsilon should decay or stay same: {epsilon_before} -> {epsilon_after}"

    def test_epsilon_independent_of_warmup(self, temp_dir):
        """Verify epsilon advances during warmup (independent of warmup)."""
        config = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            maintenance_capacity=1,
            cost_regime_id="failure-light-no-waste",
            max_steps=200,
            warmup_transitions=50,  # Large warmup
            batch_size=16,
            update_frequency=1,
            validation_interval=500,
            checkpoint_interval=100,
            replay_capacity=1000,
            training_seed=6521,
            validation_seed=6521,
            output_dir=str(temp_dir / "test_run"),
        )
        trainer = DDQNTrainer(config=config)
        obs, _ = trainer.train_env.reset()
        trainer._current_obs = obs

        epsilon_before = trainer.agent.epsilon_state.epsilon
        epsilon_step_before = trainer.agent.epsilon_state.global_step

        # Step during warmup (no gradient updates should occur)
        trainer.train_step()

        # PROOF: epsilon advanced even though still in warmup
        assert trainer.agent.epsilon_state.global_step == epsilon_step_before + 1, \
            "epsilon_state.global_step should advance even during warmup"
        assert trainer.agent.epsilon_state.epsilon <= epsilon_before, \
            "epsilon should decay (or stay same) even during warmup"

        # Verify no gradient update occurred (still in warmup)
        assert trainer.agent.gradient_update_count == 0, \
            "Should be in warmup with 0 gradient updates"

    def test_epsilon_independent_of_update_frequency(self, temp_dir):
        """Verify epsilon advances regardless of update_frequency."""
        config = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            maintenance_capacity=1,
            cost_regime_id="failure-light-no-waste",
            max_steps=200,
            warmup_transitions=10,
            batch_size=8,  # Small enough to allow updates after warmup
            update_frequency=10,  # Only update every 10 steps
            validation_interval=500,
            checkpoint_interval=100,
            replay_capacity=1000,
            training_seed=6521,
            validation_seed=6521,
            output_dir=str(temp_dir / "test_run"),
        )
        trainer = DDQNTrainer(config=config)
        obs, _ = trainer.train_env.reset()
        trainer._current_obs = obs

        # Run 10 steps
        for _ in range(10):
            trainer.train_step()

        # PROOF: epsilon advanced 10 times despite only 1 gradient update
        assert trainer.agent.epsilon_state.global_step == 10, \
            f"epsilon_state.global_step should be 10, got {trainer.agent.epsilon_state.global_step}"

        # With update_frequency=10, warmup=10, batch_size=8:
        # Steps 1-10: warmup (replay_buffer fills to 10)
        # At step 10: warmup met (10>=10), step 10 % 10 == 0, batch_size met (10>=8) = 1 update
        assert trainer.agent.gradient_update_count >= 1, \
            "Should have at least 1 gradient update after 10 steps"

    def test_gradient_update_count_counts_optimizer_updates_only(self, bounded_trainer):
        """Verify gradient_update_count counts only optimizer updates."""
        obs, _ = bounded_trainer.train_env.reset()
        bounded_trainer._current_obs = obs

        initial_grad_count = bounded_trainer.agent.gradient_update_count
        initial_step = bounded_trainer.global_step

        # Run 50 steps
        for _ in range(50):
            bounded_trainer.train_step()

        final_grad_count = bounded_trainer.agent.gradient_update_count
        final_step = bounded_trainer.global_step

        # PROOF: gradient_update_count < global_step (due to warmup)
        assert final_grad_count < final_step, \
            f"gradient_update_count ({final_grad_count}) should be < global_step ({final_step})"

        # PROOF: gradient_update_count increased
        assert final_grad_count > initial_grad_count, \
            f"gradient_update_count should have increased from {initial_grad_count}"

        # PROOF: Each gradient update corresponds to optimizer.step() call
        # The grad count should match the number of steps that passed should_update check
        # With warmup=20, update_frequency=2, batch_size=16:
        # Steps 1-20: warmup (0 updates)
        # Steps 21-50: 30 steps, update every 2 = ~15 updates
        expected_min_updates = (50 - 20) // 2
        assert final_grad_count >= expected_min_updates - 2, \
            f"gradient_update_count ({final_grad_count}) should be >= {expected_min_updates - 2}"

    def test_step_semantics_after_resume(self, temp_dir):
        """Verify step semantics remain correct after resume."""
        from training.ddqn_trainer import DDQNTrainer, TrainerConfig

        # Train trainer1
        config1 = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            maintenance_capacity=1,
            cost_regime_id="failure-light-no-waste",
            max_steps=200,
            warmup_transitions=10,
            batch_size=16,
            update_frequency=2,
            validation_interval=500,
            checkpoint_interval=50,
            replay_capacity=1000,
            training_seed=6521,
            validation_seed=6521,
            output_dir=str(temp_dir / "run1"),
        )

        trainer1 = DDQNTrainer(config=config1)
        obs, _ = trainer1.train_env.reset()
        trainer1._current_obs = obs

        for _ in range(50):
            trainer1.train_step()

        step_before_resume = trainer1.global_step
        epsilon_step_before = trainer1.agent.epsilon_state.global_step
        grad_count_before = trainer1.agent.gradient_update_count

        checkpoint_path = temp_dir / "run1" / "checkpoint_latest.pt"
        trainer1.save_checkpoint(str(checkpoint_path))

        del trainer1

        # Resume with trainer2
        config2 = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            maintenance_capacity=1,
            cost_regime_id="failure-light-no-waste",
            max_steps=200,  # MUST match trainer1 for semantic identity
            warmup_transitions=10,
            batch_size=16,
            update_frequency=2,  # MUST match trainer1
            validation_interval=500,
            checkpoint_interval=50,
            replay_capacity=1000,
            training_seed=6521,
            validation_seed=6521,
            output_dir=str(temp_dir / "run2"),
        )

        trainer2 = DDQNTrainer(config=config2, resume_from=checkpoint_path)

        # PROOF: global_step restored correctly
        assert trainer2.global_step == step_before_resume, \
            f"trainer2.global_step should be restored to {step_before_resume}"

        # PROOF: epsilon_state.global_step restored correctly
        assert trainer2.agent.epsilon_state.global_step == epsilon_step_before, \
            f"epsilon_state.global_step should be restored to {epsilon_step_before}"

        # PROOF: gradient_update_count restored correctly
        assert trainer2.agent.gradient_update_count == grad_count_before, \
            f"gradient_update_count should be restored to {grad_count_before}"

        # Continue training
        obs, _ = trainer2.train_env.reset()
        trainer2._current_obs = obs

        for _ in range(20):
            trainer2.train_step()

        # PROOF: trainer.global_step == agent.global_step after resume
        assert trainer2.global_step == trainer2.agent.global_step, \
            f"After resume: trainer.global_step={trainer2.global_step} != agent.global_step={trainer2.agent.global_step}"

        # PROOF: Steps increased correctly
        assert trainer2.global_step == step_before_resume + 20, \
            f"Expected {step_before_resume + 20} steps, got {trainer2.global_step}"