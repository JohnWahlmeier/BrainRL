#TODO: tidy up  files, review code for bugs, add environemnt state to action net observation space,
# make a visual, and test 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os
import math
import ale_py

# --- Configuration Class ---
@dataclass
class PPOConfig:
    seed: int = 42
    total_timesteps: int = 2000000
    z_dim: int = 128
    lr_actor: float = 3e-4
    lr_critic: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    ppo_epochs: int = 10
    batch_size: int = 64
    timesteps_per_batch: int = 2048
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    grad_clip: float = 0.5
    hidden_dim: int = 128
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Brain:
    def __init__(self, G, v1,v2, config, env_name):
        self.cfg = config
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]

        self.z_dim = config.z_dim
        self.n = np.shape(G)[0]
        
        # Build graph topology
        self.incoming_edges = [[] for _ in range(self.n)]
        for j in range(self.n):
            for i in range(self.n):
                if G[i,j] == 1:
                    self.incoming_edges[j].append(i)

        self.is_input_node = [1 if v1[i]=='input_node' else 0 for i in range(self.n)]
        self.is_output_node = [1 if v2[i]=='output_node' else 0 for i in range(self.n)]
        
        # Calculate input sizes for each node's internal PPO agent
        self.V_input_sizes = [
            len(self.incoming_edges[i]) * self.z_dim + (self.obs_dim if self.is_input_node[i] else 0) 
            for i in range(self.n)
        ]
        
        # Initialize internal nodes
        self.V = [ContinuousPPOAgent(state_size=self.V_input_sizes[i], action_size=self.z_dim, config=config) 
                  for i in range(self.n)]
        
        # Output nodes feed into the final action network
        self.action_vertices = [i for i, val in enumerate(self.is_output_node) if val == 1]
        self.action_net = ContinuousPPOAgent(state_size=self.z_dim * len(self.action_vertices), 
                                            action_size=self.act_dim, config=config)
        
        # Latent state "activations"
        self.z = torch.zeros((self.n, self.z_dim), device=self.cfg.device)
        self.z_buffer = torch.zeros((self.n, self.z_dim), device=self.cfg.device)

    def _get_node_state(self, i, obs_t):
        """Helper to assemble the state for node i"""
        incoming_z = self.z[self.incoming_edges[i]].view(-1) # Flatten incoming signals
        if self.is_input_node[i]:
            return torch.cat([obs_t, incoming_z]).unsqueeze(0)
        return incoming_z.unsqueeze(0)

    def train(self):
        print(f"Training Brain on {self.env_name}")
        history = {'steps': [], 'rewards': []}
        total_steps = 0
        
        while total_steps < self.cfg.total_timesteps:
            # Reset buffers
            V_buffers = [{'states': [], 'actions': [], 'log_probs': [], 'values': []} for _ in range(self.n)]
            action_net_buffer = {'states': [], 'actions': [], 'log_probs': [], 'values': []}
            rewards, dones = [], []

            state, _ = self.env.reset()
            self.z = torch.zeros((self.n, self.z_dim), device=self.cfg.device)
            
            # --- Collection Loop ---
            for t in range(self.cfg.timesteps_per_batch):
                obs_t = torch.from_numpy(state).float().to(self.cfg.device)
                
                with torch.no_grad():
                    # Update each internal node
                    for i in range(self.n):
                        state_i = self._get_node_state(i, obs_t)
                        dist = self.V[i].get_dist(state_i)
                        val = self.V[i].vnetwork(state_i)
                        
                        new_z = dist.sample()
                        logprob = dist.log_prob(new_z).sum(dim=-1)
                        
                        self.z_buffer[i] = new_z.squeeze(0)
                        V_buffers[i]['states'].append(state_i.cpu().numpy()[0])
                        V_buffers[i]['actions'].append(new_z.cpu().numpy()[0])
                        V_buffers[i]['log_probs'].append(logprob.item())
                        V_buffers[i]['values'].append(val.item())

                    # Final Action Net
                    state_act = self.z[self.action_vertices].view(1, -1)
                    dist_act = self.action_net.get_dist(state_act)
                    val_act = self.action_net.vnetwork(state_act)
                    
                    action = dist_act.sample()
                    logprob_act = dist_act.log_prob(action).sum(dim=-1)
                    action_np = action.cpu().numpy()[0]

                    action_net_buffer['states'].append(state_act.cpu().numpy()[0])
                    action_net_buffer['actions'].append(action_np)
                    action_net_buffer['log_probs'].append(logprob_act.item())
                    action_net_buffer['values'].append(val_act.item())

                next_state, reward, terminated, truncated, _ = self.env.step(action_np)
                done = terminated or truncated
                
                rewards.append(reward)
                dones.append(done)
                state = next_state
                self.z = self.z_buffer.clone().detach()

                if done:
                    state, _ = self.env.reset()
                    self.z = torch.zeros((self.n, self.z_dim), device=self.cfg.device)

            total_steps += self.cfg.timesteps_per_batch

            # --- Update Phase ---
            with torch.no_grad():
                obs_t = torch.from_numpy(state).float().to(self.cfg.device)
                last_vals_i = [self.V[i].vnetwork(self._get_node_state(i, obs_t)).item() for i in range(self.n)]
                state_act_last = self.z[self.action_vertices].view(1, -1)
                last_val_act = self.action_net.vnetwork(state_act_last).item()

            # Update nodes
            for i in range(self.n):
                adv, ret = self.compute_gae(rewards, V_buffers[i]['values'], dones, last_vals_i[i])
                self.V[i].update(
                    torch.tensor(np.array(V_buffers[i]['states']), dtype=torch.float32, device=self.cfg.device),
                    torch.tensor(np.array(V_buffers[i]['actions']), dtype=torch.float32, device=self.cfg.device),
                    torch.tensor(np.array(V_buffers[i]['log_probs']), dtype=torch.float32, device=self.cfg.device),
                    ret, adv
                )

            # Update action net
            adv_a, ret_a = self.compute_gae(rewards, action_net_buffer['values'], dones, last_val_act)
            self.action_net.update(
                torch.tensor(np.array(action_net_buffer['states']), dtype=torch.float32, device=self.cfg.device),
                torch.tensor(np.array(action_net_buffer['actions']), dtype=torch.float32, device=self.cfg.device),
                torch.tensor(np.array(action_net_buffer['log_probs']), dtype=torch.float32, device=self.cfg.device),
                ret_a, adv_a
            )

            # Logging
            eval_reward = self.evaluate()
            history['steps'].append(total_steps)
            history['rewards'].append(eval_reward)
            print(f"Step: {total_steps} | Eval Reward: {eval_reward:.2f}")

        return history

    def compute_gae(self, rewards, values, dones, last_value):
        rewards = np.array(rewards, dtype=np.float32)
        values = np.array(values + [last_value], dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * values[t+1] * mask - values[t]
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * mask * last_gae
            advantages[t] = last_gae
            
        returns = advantages + values[:-1]
        adv_tensor = torch.tensor(advantages, device=self.cfg.device)
        ret_tensor = torch.tensor(returns, device=self.cfg.device)
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
        return adv_tensor, ret_tensor

    def evaluate(self, episodes=3):
        total_reward = 0
        for _ in range(episodes):
            s, _ = self.env.reset()
            z_eval = torch.zeros((self.n, self.z_dim), device=self.cfg.device)
            done = False
            while not done:
                obs_t = torch.from_numpy(s).float().to(self.cfg.device)
                with torch.no_grad():
                    # Propagation
                    new_z_eval = torch.zeros_like(z_eval)
                    for i in range(self.n):
                        # Use z_eval here to get current step's node state
                        incoming_z = z_eval[self.incoming_edges[i]].view(-1)
                        st = torch.cat([obs_t, incoming_z]) if self.is_input_node[i] else incoming_z
                        new_z_eval[i] = self.V[i].pnetwork.forward(st.unsqueeze(0))[0].squeeze(0)
                    
                    st_act = new_z_eval[self.action_vertices].view(1, -1)
                    act, _ = self.action_net.pnetwork.forward(st_act)
                    s, r, term, trunc, _ = self.env.step(act.cpu().numpy()[0])
                    z_eval = new_z_eval
                    total_reward += r
                    done = term or trunc
        return total_reward / episodes

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class ContinuousPolicyNetwork(nn.Module):
    def __init__(self, s_size, z_size, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(s_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, z_size)
        self.log_std = nn.Parameter(torch.zeros(z_size))

    def forward(self, state):
        x = self.fc(state)
        mu = torch.tanh(self.fc_mu(x))
        return mu, self.log_std

    def get_dist(self, state):
        mu, log_std = self.forward(state)
        return Normal(mu, log_std.exp().expand_as(mu))

class ContinuousPPOAgent:
    def __init__(self, state_size, action_size, config: PPOConfig):
        self.cfg = config
        self.pnetwork = ContinuousPolicyNetwork(state_size, action_size, self.cfg.hidden_dim).to(self.cfg.device)
        self.vnetwork = ValueNetwork(state_size, self.cfg.hidden_dim).to(self.cfg.device)
        self.poptimizer = optim.Adam(self.pnetwork.parameters(), lr=self.cfg.lr_actor)
        self.voptimizer = optim.Adam(self.vnetwork.parameters(), lr=self.cfg.lr_critic)

    def get_dist(self, state):
        return self.pnetwork.get_dist(state)

    def update(self, states, actions, old_log_probs, returns, advantages):
        for _ in range(self.cfg.ppo_epochs):
            dist = self.pnetwork.get_dist(states)
            new_logp = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            ratio = torch.exp(new_logp - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean() - self.cfg.entropy_coeff * entropy
            value_loss = F.mse_loss(self.vnetwork(states).squeeze(-1), returns)

            self.poptimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.pnetwork.parameters(), self.cfg.grad_clip)
            self.poptimizer.step()

            self.voptimizer.zero_grad()
            (self.cfg.value_coeff * value_loss).backward()
            nn.utils.clip_grad_norm_(self.vnetwork.parameters(), self.cfg.grad_clip)
            self.voptimizer.step()

# --- Example Usage Fix ---
if __name__ == '__main__':
    cfg = PPOConfig()
    # Simple graph: 1 input node -> 1 output node
    G = np.array([[0]])
    v1 = ['input_node']
    v2 = ['output_node']
    
    brain = Brain(G, v1, v2, cfg, "Ant-v5")
    brain.train()
