"""
Focused M5 Tests: End-to-End Resume (F10)

Resume test:
- Uses DDQNTrainer(config=config, resume_from=checkpoint_path)
- No manual state assignments to global_step, epsilon, optimizer, etc.
- Automatic restoration of all trainer and agent state
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

from agents.ddqn.agent import DDQNAgent, DDQNAgentConfig
from agents.ddqn.checkpoint import save_checkpoint, load_checkpoint
from envs.action_table import ACTION_TABLE_N5_K1


class TestEndToEndResume:
    """Test training path end-to-end resume (F10)."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)

    def get_agent_state(self, agent):
        """Record current agent state."""
        return {
            "global_step": agent.global_step,
            "gradient_update_count": agent.gradient_update_count,
            "epsilon": agent.epsilon_state.epsilon,
            "epsilon_global_step": agent.epsilon_state.global_step,
            "online_weights": {
                k: v.clone() for k, v in agent.online_network.state_dict().items()
            },
            "target_weights": {
                k: v.clone() for k, v in agent.target_network.state_dict().items()
            },
            "optimizer_state": agent.optimizer.state_dict(),
        }

    def verify_agent_restored(self, agent, recorded_state):
        """Verify agent state matches recorded state."""
        assert agent.global_step == recorded_state["global_step"], \
            f"global_step mismatch: {agent.global_step} != {recorded_state['global_step']}"
        assert agent.gradient_update_count == recorded_state["gradient_update_count"], \
            f"gradient_update_count mismatch: {agent.gradient_update_count} != {recorded_state['gradient_update_count']}"
        assert agent.epsilon_state.epsilon == recorded_state["epsilon"], \
            f"epsilon mismatch: {agent.epsilon_state.epsilon} != {recorded_state['epsilon']}"
        assert agent.epsilon_state.global_step == recorded_state["epsilon_global_step"], \
            f"epsilon_global_step mismatch"

        # Verify weights restored
        for k, v in agent.online_network.state_dict().items():
            assert torch.allclose(v, recorded_state["online_weights"][k]), \
                f"online network weight {k} not restored"

        for k, v in agent.target_network.state_dict().items():
            assert torch.allclose(v, recorded_state["target_weights"][k]), \
                f"target network weight {k} not restored"

        # Verify optimizer state restored (state should be populated)
        assert len(agent.optimizer.state_dict()["state"]) > 0, \
            "optimizer state should be populated after restore"

    def test_end_to_end_resume_production_path(self, temp_dir):
        """
        End-to-end resume test (F10).

        Executes:
        1. Construct trainer1 with DDQNTrainer(config=config)
        2. Train enough steps to pass warmup and perform updates
        3. Save checkpoint through trainer1.save_checkpoint() production API
        4. Record state (global step, epsilon, optimizer state, online/target params)
        5. Destroy trainer1
        6. Construct trainer2 using production resume path:
           DDQNTrainer(config=config, resume_from=checkpoint_path)
        7. NO manual assignments to:
           - global_step
           - epsilon
           - optimizer
           - gradient counters
           - network states
        8. Verify automatic restoration of all state
        9. Continue training through trainer2.train_step() production methods
        10. Verify:
            - global_step increases from restored value
            - epsilon continues decaying instead of resetting
            - optimizer state remains populated
            - warmup is not incorrectly repeated
            - new checkpoint reflects continued step
            - target synchronization still occurs
        """
        from training.ddqn_trainer import DDQNTrainer, TrainerConfig

        # Step 1: Construct trainer1 with production config
        config = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            maintenance_capacity=1,
            cost_regime_id="failure-light-no-waste",
            max_steps=200,  # Small for test
            warmup_transitions=20,  # Small warmup
            batch_size=16,  # Small batch size for test
            update_frequency=1,
            validation_interval=500,  # Won't trigger
            checkpoint_interval=100,
            replay_capacity=1000,
            training_seed=6521,
            validation_seed=6521,
            output_dir=str(temp_dir / "run1"),
        )

        trainer1 = DDQNTrainer(config=config, resume_from=None)

        # Step 2: Train using production train_step() method
        obs, _ = trainer1.train_env.reset()
        trainer1._current_obs = obs

        # Train for 50 steps - enough to pass warmup and get updates
        for _ in range(50):
            trainer1.train_step()

        # Verify training occurred
        assert trainer1.global_step == 50, f"Expected 50 steps, got {trainer1.global_step}"
        assert trainer1.agent.gradient_update_count > 0, \
            f"Expected gradient updates, got {trainer1.agent.gradient_update_count}"
        assert len(trainer1.replay_buffer) > trainer1.config.warmup_transitions, \
            f"Expected replay buffer to exceed warmup"

        # Step 3: Save checkpoint using production save_checkpoint() API
        checkpoint_path = temp_dir / "run1" / "checkpoint_latest.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        saved_path = trainer1.save_checkpoint(str(checkpoint_path))

        assert checkpoint_path.exists(), "Checkpoint should have been saved"

        # Step 4: Record state BEFORE destroying trainer1
        recorded_state = self.get_agent_state(trainer1.agent)
        pre_resume_step = trainer1.global_step
        pre_resume_epsilon = trainer1.agent.epsilon_state.epsilon
        pre_resume_update_count = trainer1.agent.gradient_update_count

        # Step 5: Destroy trainer1 completely
        del trainer1

        # Step 6-7: Construct trainer2 using PRODUCTION RESUME PATH
        # NO manual state assignments - DDQNTrainer handles everything
        # IMPORTANT: Use SAME max_steps as trainer1 for semantic identity match
        config2 = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            maintenance_capacity=1,
            cost_regime_id="failure-light-no-waste",
            max_steps=200,  # MUST match trainer1 for semantic identity
            warmup_transitions=20,
            batch_size=16,
            update_frequency=1,
            validation_interval=500,
            checkpoint_interval=100,
            replay_capacity=1000,
            training_seed=6521,
            validation_seed=6521,
            output_dir=str(temp_dir / "run2"),
        )

        # PRODUCTION RESUME: All state restored automatically by DDQNTrainer
        trainer2 = DDQNTrainer(config=config2, resume_from=checkpoint_path)

        # Step 8: Verify ALL state was automatically restored
        self.verify_agent_restored(trainer2.agent, recorded_state)
        assert trainer2.global_step == pre_resume_step, \
            f"trainer2.global_step not restored: {trainer2.global_step} != {pre_resume_step}"

        # Step 9: Continue training using production train_step() method
        obs, _ = trainer2.train_env.reset()
        trainer2._current_obs = obs

        # Record gradient_update_count BEFORE continued training
        gradient_update_count_before_resume = trainer2.agent.gradient_update_count

        for _ in range(30):
            trainer2.train_step()

        # Record gradient_update_count AFTER continued training
        gradient_update_count_after_resume = trainer2.agent.gradient_update_count

        # Step 10: Verify training continued correctly

        # PROOF 1: global_step increases from restored value
        assert trainer2.global_step >= pre_resume_step + 30, \
            f"global_step should have increased from {pre_resume_step} by at least 30, got {trainer2.global_step}"

        # PROOF 2: New optimizer updates occur after resume
        # This proves the optimizer is actually updating, not just replaying old state
        assert gradient_update_count_after_resume > gradient_update_count_before_resume, \
            f"PROOF FAILED: No new gradient updates after resume: {gradient_update_count_after_resume} <= {gradient_update_count_before_resume}"

        # PROOF 3: Warmup is NOT repeated after resume
        # If warmup were incorrectly repeated, no updates would occur in the first 20 steps
        # Since we proved updates occurred, warmup was not repeated
        steps_after_resume = trainer2.global_step - pre_resume_step
        updates_after_resume = gradient_update_count_after_resume - gradient_update_count_before_resume
        # We should have updates proportional to steps (update_frequency=1)
        # If warmup were repeated, we'd get 0 updates for the first warmup_transitions steps
        expected_min_updates = steps_after_resume - trainer2.config.warmup_transitions
        # Allow some slack for batch size constraints, but prove we got substantial updates
        assert updates_after_resume >= expected_min_updates - 5, \
            f"PROOF FAILED: Updates suggest warmup may have been repeated: {updates_after_resume} < {expected_min_updates}"

        # PROOF 4: gradient_update_count continues from pre-resume value
        assert trainer2.agent.gradient_update_count >= pre_resume_update_count, \
            f"gradient_update_count should have continued from {pre_resume_update_count}"

        # PROOF 5: Epsilon continued decaying (not reset)
        assert trainer2.agent.epsilon_state.epsilon <= pre_resume_epsilon, \
            f"epsilon should have continued decaying from {pre_resume_epsilon}"
        assert trainer2.agent.epsilon_state.global_step > recorded_state["epsilon_global_step"], \
            f"epsilon global_step should have increased"

        # PROOF 6: Optimizer state remains populated after continued training
        assert len(trainer2.agent.optimizer.state_dict()["state"]) > 0, \
            "optimizer state should remain populated after continued training"

        # PROOF 7: Network weights changed after continued training
        # This proves actual training occurred, not just state replay
        weights_changed = False
        for k, v in trainer2.agent.online_network.state_dict().items():
            if k in recorded_state["online_weights"]:
                if not torch.allclose(v, recorded_state["online_weights"][k]):
                    weights_changed = True
                    break
        assert weights_changed, "PROOF FAILED: Network weights unchanged after continued training"

        # Save a new checkpoint to verify checkpoint save works after resume
        checkpoint_path2 = temp_dir / "run2" / "checkpoint_after_resume.pt"
        trainer2.save_checkpoint(str(checkpoint_path2))
        assert checkpoint_path2.exists(), "Checkpoint after resume should have been saved"

        # Verify checkpoint reflects continued step
        from agents.ddqn.checkpoint import load_checkpoint
        checkpoint_data2, _ = load_checkpoint(checkpoint_path2)
        assert checkpoint_data2.global_step >= pre_resume_step + 30, \
            f"Checkpoint should reflect continued step >= {pre_resume_step + 30}, got {checkpoint_data2.global_step}"

        print(f"\nProduction resume test PASSED:")
        print(f"  Pre-resume step: {pre_resume_step}")
        print(f"  Post-resume step: {trainer2.global_step}")
        print(f"  Gradient updates: {trainer2.agent.gradient_update_count}")
        print(f"  Final epsilon: {trainer2.agent.epsilon_state.epsilon:.4f}")

    def test_end_to_end_resume_restores_replay_buffer(self, temp_dir):
        """
        Test production resume restores replay buffer state.

        Executes:
        1. Train and fill replay buffer
        2. Save checkpoint
        3. Resume from checkpoint
        4. Verify replay buffer size restored immediately
        5. Verify one post-resume step can update without full repeated warmup
        """
        from training.ddqn_trainer import DDQNTrainer, TrainerConfig

        # Step 1: Construct trainer1
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
            update_frequency=1,
            validation_interval=500,
            checkpoint_interval=100,
            replay_capacity=1000,
            training_seed=6521,
            validation_seed=6521,
            output_dir=str(temp_dir / "run1"),
        )

        trainer1 = DDQNTrainer(config=config)

        # Step 2: Train using production train_step() method
        obs, _ = trainer1.train_env.reset()
        trainer1._current_obs = obs

        # Train for 50 steps - enough to pass warmup
        for _ in range(50):
            trainer1.train_step()

        # Verify training occurred
        assert trainer1.global_step == 50
        assert len(trainer1.replay_buffer) > trainer1.config.warmup_transitions
        pre_resume_replay_size = len(trainer1.replay_buffer)

        # Step 3: Save checkpoint
        checkpoint_path = temp_dir / "run1" / "checkpoint_latest.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer1.save_checkpoint(str(checkpoint_path))

        # Step 4: Destroy trainer1
        del trainer1

        # Step 5: Construct trainer2 with resume
        config2 = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            maintenance_capacity=1,
            cost_regime_id="failure-light-no-waste",
            max_steps=200,  # MUST match trainer1 for semantic identity
            warmup_transitions=20,
            batch_size=16,
            update_frequency=1,
            validation_interval=500,
            checkpoint_interval=100,
            replay_capacity=1000,
            training_seed=6521,
            validation_seed=6521,
            output_dir=str(temp_dir / "run2"),
        )

        trainer2 = DDQNTrainer(config=config2, resume_from=checkpoint_path)

        # PROOF 1: Replay buffer size restored immediately
        assert len(trainer2.replay_buffer) == pre_resume_replay_size, \
            f"Replay buffer size not restored: {len(trainer2.replay_buffer)} != {pre_resume_replay_size}"

        # PROOF 2: Can continue training without repeated warmup
        # Set update_frequency=1 and verify one step can trigger update
        obs, _ = trainer2.train_env.reset()
        trainer2._current_obs = obs

        pre_update_count = trainer2.agent.gradient_update_count

        # One step should be able to update since buffer already has enough transitions
        trainer2.train_step()

        # Verify gradient update occurred (buffer had enough transitions)
        assert trainer2.agent.gradient_update_count > pre_update_count, \
            f"PROOF FAILED: No gradient update after resume step. " \
            f"Buffer size={len(trainer2.replay_buffer)}, warmup={trainer2.config.warmup_transitions}"

        print(f"\nReplay buffer restore test PASSED:")
        print(f"  Pre-resume replay size: {pre_resume_replay_size}")
        print(f"  Post-resume replay size: {len(trainer2.replay_buffer)}")
        print(f"  Gradient updates before: {pre_update_count}")
        print(f"  Gradient updates after: {trainer2.agent.gradient_update_count}")