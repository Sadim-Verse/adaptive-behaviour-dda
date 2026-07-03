import torch as th
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.buffers import ReplayBuffer

class DDQN(DQN):
    """
    Double Deep Q-Network (DDQN) implementation.
    Extends SB3's DQN by using the Double DQN target update.
    """
    def train(self, gradient_steps: int, batch_size: int) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # 1. Q-values from target network (all actions)
                next_q_values = self.q_net_target(replay_data.next_observations)  # (64, 3)
                
                # 2. Select actions using ONLINE network
                next_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)  # (64, 1)
                
                # 3. Evaluate those actions using TARGET network
                target_q_values = next_q_values.gather(1, next_actions)  # (64, 1)
                
                # 4. TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q_values  # (64, 1)

            # Current Q-values (from online network, for the actions taken)
            current_q_values = self.q_net(replay_data.observations).gather(1, replay_data.actions)  # (64, 1)

            # Loss
            loss = th.nn.functional.smooth_l1_loss(current_q_values, target_q_values)

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            # CRITICAL FIX: Increment the update counter so the target network actually updates
            self._n_updates += 1

            # Update target network inside the gradient steps loop
            if self.target_update_interval > 0 and self._n_updates % self.target_update_interval == 0:
                polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)