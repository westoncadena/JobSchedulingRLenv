# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) inform Guni Sharon at 
# guni@tamu.edu regarding your usage (relevant statistics is reported to NSF).
# The development of this assignment was supported by NSF (IIS-2238979).
# Contributors:
# The core code base was developed by Guni Sharon (guni@tamu.edu).
# The PyTorch code was developed by Sheelabhadra Dey (sheelabhadra@tamu.edu).

import random
from copy import deepcopy
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW

from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting

class QNetwork(nn.Module):
    """
    Neural network to approximate Q-values.
    """
    def __init__(self, input_dim, output_dim, hidden_sizes):
        super(QNetwork, self).__init__()
        layers = [nn.Linear(input_dim, hidden_sizes[0]), nn.ReLU()]
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DQN(AbstractSolver):
    def __init__(self, env, eval_env, options):
        super().__init__(env, eval_env, options)

        # Process observation space
        # Flatten the Dict observation space
        self.input_dim = (
            np.prod(env.observation_space["node_resources"].shape) +
            np.prod(env.observation_space["job_queue"].shape)
        )
        
        # Calculate total number of possible actions
        # For your case: num_actions = action_types * job_indices * node_indices * delay_steps
        self.output_dim = (
            env.action_space["action_type"].n *
            env.action_space["job_index"].n *
            env.action_space["node_index"].n *
            env.action_space["delay_steps"].n
        )

        # Initialize Q-network and target network
        self.model = QNetwork(self.input_dim, self.output_dim, options["hidden_sizes"])
        self.target_model = deepcopy(self.model)
        self.optimizer = AdamW(self.model.parameters(), lr=options["lr"])
        self.loss_fn = nn.SmoothL1Loss()

        # Replay buffer
        self.replay_memory = deque(maxlen=options["replay_memory_size"])
        self.batch_size = options["batch_size"]

        # Training parameters
        self.gamma = options["gamma"]
        self.epsilon = options["epsilon"]
        self.epsilon_decay = options["epsilon_decay"]
        self.min_epsilon = options["min_epsilon"]
        self.update_target_every = options["update_target_every"]
        self.steps = 0

    def _process_state(self, state):
        """Convert dict state to flat array"""
        
        node_resources = state["node_resources"].flatten()
        job_queue = state["job_queue"].flatten()
        return np.concatenate([node_resources, job_queue])

    def _decode_action(self, action_idx):
        """Convert flat action index to dict action"""
        action_type_n = self.env.action_space["action_type"].n
        job_index_n = self.env.action_space["job_index"].n
        node_index_n = self.env.action_space["node_index"].n
        delay_steps_n = self.env.action_space["delay_steps"].n

        # Decode the flat index into individual components
        delay_steps = action_idx % delay_steps_n
        action_idx //= delay_steps_n
        
        node_index = action_idx % node_index_n
        action_idx //= node_index_n
        
        job_index = action_idx % job_index_n
        action_type = action_idx // job_index_n

        return {
            "action_type": int(action_type),
            "job_index": int(job_index),
            "node_index": int(node_index),
            "delay_steps": int(delay_steps)
        }

    def epsilon_greedy_policy(self, state):
        """
        Epsilon-greedy policy for action selection.
        """
        if np.random.rand() < self.epsilon:
            # Random action
            action_type = self.env.action_space["action_type"].sample()
            job_index = self.env.action_space["job_index"].sample()
            node_index = self.env.action_space["node_index"].sample()
            delay_steps = self.env.action_space["delay_steps"].sample()
            return {
                "action_type": action_type,
                "job_index": job_index,
                "node_index": node_index,
                "delay_steps": delay_steps
            }
        else:
            flattened_state = self._process_state(state)
            state_tensor = torch.tensor(self._process_state(state), 
                                      dtype=torch.float32).unsqueeze(0)
            q_values = self.model(state_tensor)
            action_idx = torch.argmax(q_values).item()
            return self._decode_action(action_idx)

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition in replay memory.
        """
        # Convert state and action to flat representations
        flat_state = self._process_state(state)
        
        # Convert dict action to flat index
        action_type = action["action_type"]
        job_index = action["job_index"]
        node_index = action["node_index"]
        delay_steps = action["delay_steps"]
        
        flat_action = (action_type * 
                      self.env.action_space["job_index"].n * 
                      self.env.action_space["node_index"].n * 
                      self.env.action_space["delay_steps"].n +
                      job_index * 
                      self.env.action_space["node_index"].n * 
                      self.env.action_space["delay_steps"].n +
                      node_index * 
                      self.env.action_space["delay_steps"].n +
                      delay_steps)
        
        flat_next_state = self._process_state(next_state)
        
        self.replay_memory.append((flat_state, flat_action, reward, flat_next_state, done))

    def replay(self):
        """
        Update Q-network using experiences from the replay buffer.
        """
        if len(self.replay_memory) < self.batch_size:
            return

        # Sample minibatch
        batch = random.sample(self.replay_memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Process tensors
        states = torch.tensor(np.array(states), dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)

        # Q-value updates
        current_q_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(current_q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

    def update_target_model(self):
        """
        Update target network weights.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def train_episode(self):
        """
        Train for one episode.
        """
        state, _ = self.env.reset()  # Ensure this returns the correct state format

        total_reward = 0

        while True:
            action = self.epsilon_greedy_policy(state)
            next_state, reward, done, _ = self.step(action)  
            self.store_transition(state, action, reward, next_state, done)

            self.replay()
            total_reward += reward
            state = next_state

            if done:
                break

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        # Update target network periodically
        if self.steps % self.update_target_every == 0:
            self.update_target_model()

        return total_reward

    def __str__(self):
        return "DQN"

    def create_greedy_policy(self):
        """
        Creates a greedy policy based on Q values.
        """
        def policy_fn(state):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()

        return policy_fn
