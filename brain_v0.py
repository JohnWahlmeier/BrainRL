#TODO change eval
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os
import math
import ale_py
import random

# --- Configuration Class ---
@dataclass
class PPOConfig:
    seed: int = 42
    total_timesteps: int =1000000
    z_dim: int = 4 
    lr_actor: float = 3e-4
    lr_critic: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.8
    clip_epsilon: float = 0.1
    ppo_epochs: int = 5
    batch_size: int = 64
    timesteps_per_batch: int = 2048
    value_coeff: float = 0.5
    entropy_coeff: float = 0.001
    grad_clip = 0.5
    
    hidden_dim: int = 256
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Shared Network Components ---

class ValueNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- Agent A Components (Continuous) ---

class ContinuousPolicyNetwork(nn.Module):
    def __init__(self, s_size, z_size, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(s_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, z_size)
        self.log_std = nn.Parameter(torch.zeros(z_size))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) 
        return mu, self.log_std

    def get_dist(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp().expand_as(mu)
        return Normal(mu, std)

class ContinuousPPOAgent:
    def __init__(self, state_size, action_size, config: PPOConfig):
        self.cfg = config
        self.pnetwork = ContinuousPolicyNetwork(state_size, action_size, self.cfg.hidden_dim).to(self.cfg.device)
        self.vnetwork = ValueNetwork(state_size, self.cfg.hidden_dim).to(self.cfg.device)
        self.poptimizer = optim.Adam(self.pnetwork.parameters(), lr=self.cfg.lr_actor)
        self.voptimizer = optim.Adam(self.vnetwork.parameters(), lr=self.cfg.lr_critic)

    def update(self, states, actions, old_log_probs, returns, advantages):
        N = states.shape[0]
        batch_size = self.cfg.batch_size
        
        for _ in range(self.cfg.ppo_epochs):
            perm = torch.randperm(N)
            for start in range(0, N, batch_size):
                idx = perm[start:start + batch_size]
                b_states, b_actions = states[idx], actions[idx]
                b_oldlogp, b_adv, b_ret = old_log_probs[idx], advantages[idx], returns[idx]

                dist = self.pnetwork.get_dist(b_states)
                
                new_logp = dist.log_prob(b_actions).sum(dim=-1) 
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(new_logp - b_oldlogp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred = self.vnetwork(b_states).squeeze(-1)
                value_loss = F.mse_loss(value_pred, b_ret)

                self.poptimizer.zero_grad()
                (policy_loss - self.cfg.entropy_coeff * entropy).backward()
                nn.utils.clip_grad_norm_(self.pnetwork.parameters(), self.cfg.grad_clip)
                self.poptimizer.step()

                self.voptimizer.zero_grad()
                (self.cfg.value_coeff * value_loss).backward()
                nn.utils.clip_grad_norm_(self.vnetwork.parameters(), self.cfg.grad_clip)
                self.voptimizer.step()

# --- Agent B Components (Discrete) ---

class DiscretePolicyNetwork(nn.Module):

    def __init__(self, input_dim, action_size, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def get_dist(self, x):
        logits = self.forward(x)
        return Categorical(logits=logits)

class DiscretePPOAgent:

    def __init__(self, input_dim, action_size, config: PPOConfig):
        self.cfg = config
        self.pnetwork = DiscretePolicyNetwork(input_dim, action_size, self.cfg.hidden_dim).to(self.cfg.device)
        self.vnetwork = ValueNetwork(input_dim, self.cfg.hidden_dim).to(self.cfg.device)
        self.poptimizer = optim.Adam(self.pnetwork.parameters(), lr=self.cfg.lr_actor)
        self.voptimizer = optim.Adam(self.vnetwork.parameters(), lr=self.cfg.lr_critic)

    def update(self, states, actions, old_log_probs, returns, advantages):
        N = states.shape[0]
        batch_size = self.cfg.batch_size

        for _ in range(self.cfg.ppo_epochs):
            perm = torch.randperm(N)
            for start in range(0, N, batch_size):
                idx = perm[start:start + batch_size]
                b_states, b_actions = states[idx], actions[idx]
                b_oldlogp, b_adv, b_ret = old_log_probs[idx], advantages[idx], returns[idx]

                dist = self.pnetwork.get_dist(b_states)
                new_logp = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - b_oldlogp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred = self.vnetwork(b_states).squeeze(-1)
                value_loss = F.mse_loss(value_pred, b_ret)

                self.poptimizer.zero_grad()
                (policy_loss - self.cfg.entropy_coeff * entropy).backward()
                nn.utils.clip_grad_norm_(self.pnetwork.parameters(), self.cfg.grad_clip)
                self.poptimizer.step()

                self.voptimizer.zero_grad()
                (self.cfg.value_coeff * value_loss).backward()
                nn.utils.clip_grad_norm_(self.vnetwork.parameters(), self.cfg.grad_clip)
                self.voptimizer.step()

# --- Global Trainer ---

class GlobalTrainer_Multi:
    def __init__(self, config: PPOConfig, env_name):
        self.cfg = config
        def make_env(is_training=True):
            if env_name in ['ALE/Frogger-v5']:
                env = gym.make(env_name, obs_type="ram" , frameskip=1, repeat_action_probability=0)
            else:
                env = gym.make(env_name)
            return env
        
        self.env = make_env(is_training=True)
        self.eval_env = make_env(is_training=False)
        self.obs_dim = self.env.observation_space.shape[0]
        
        
        self.is_continuous = isinstance(self.env.action_space, gym.spaces.Box)
        
        if self.is_continuous:
            act_dim = self.env.action_space.shape[0]
            self.agent_B = ContinuousPPOAgent(self.obs_dim + self.cfg.z_dim, act_dim, self.cfg)
        else:
            act_dim = self.env.action_space.n 
            self.agent_B = DiscretePPOAgent(self.obs_dim + self.cfg.z_dim, act_dim, self.cfg)
        
        self.agent_A = ContinuousPPOAgent(self.obs_dim, self.cfg.z_dim, self.cfg)

    def compute_gae(self, rewards, values, dones, last_value):
        N = len(rewards)
        advantages = np.zeros(N, dtype=np.float32)
        last_gae = 0.0
        
        for t in reversed(range(N)):
            if t == N - 1:
                next_value = last_value
                next_nonterminal = 0.0 if dones[t] else 1.0
            else:
                next_value = values[t + 1]
                next_nonterminal = 0.0 if dones[t + 1] else 1.0

            delta = rewards[t] + self.cfg.gamma * next_value * next_nonterminal - values[t]
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(values, dtype=np.float32)
        
        adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.cfg.device)
        ret_tensor = torch.tensor(returns, dtype=torch.float32, device=self.cfg.device)
        
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
        return adv_tensor, ret_tensor

    def train(self,env_name):
        print(f"Training Hierarchical Agents on {env_name}")
        total_steps = 0
        history = {'steps': [], 'rewards': []}
        z_ts = []
        o_ts = []

        try:
            while total_steps < self.cfg.total_timesteps:
                buf_A = {'states': [], 'actions': [], 'log_probs': [], 'values': []}
                buf_B = {'states': [], 'actions': [], 'log_probs': [], 'values': []}
                common = {'rewards': [], 'dones': []}
                
                state, _ = self.env.reset()
                random_integer = random.randrange(self.cfg.timesteps_per_batch)
                
                for t in range(self.cfg.timesteps_per_batch):

                    state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.cfg.device)
                    
                    with torch.no_grad():
                        dist_a = self.agent_A.pnetwork.get_dist(state_t)
                        z_t = dist_a.sample()
                        logprob_a = dist_a.log_prob(z_t).sum(dim=-1) # Sum logprobs for vector
                        val_a = self.agent_A.vnetwork(state_t)

                    z_np = z_t.cpu().numpy()
                    if t == random_integer:
                        z_ts.append(z_np)
                        o_ts.append(state)


                    combined_input_t = torch.cat([state_t, z_t], dim=1)
                    
                    with torch.no_grad():
                        dist_b = self.agent_B.pnetwork.get_dist(combined_input_t)
                        action_t = dist_b.sample()
                        val_b = self.agent_B.vnetwork(combined_input_t)

                        if self.is_continuous:
                            logprob_b = dist_b.log_prob(action_t).sum(dim=-1)
                            env_action = action_t.cpu().numpy()[0] 
                            buf_action = action_t.cpu().numpy()[0]
                        else:
                            logprob_b = dist_b.log_prob(action_t)
                            env_action = action_t.item()
                            buf_action = action_t.item()
                    
                    next_state, reward, terminated, truncated, _ = self.env.step(env_action)
                    done = terminated or truncated

                    buf_A['states'].append(state)
                    buf_A['actions'].append(z_np[0])
                    buf_A['log_probs'].append(logprob_a.item())
                    buf_A['values'].append(val_a.item())

                    combined_np = combined_input_t.cpu().numpy()[0]
                    buf_B['states'].append(combined_np)
                    buf_B['actions'].append(buf_action)
                    buf_B['log_probs'].append(logprob_b.item())
                    buf_B['values'].append(val_b.item())

                    common['rewards'].append(reward)
                    common['dones'].append(done)

                    state = next_state
                    if done:
                        state, _ = self.env.reset()

                total_steps += self.cfg.timesteps_per_batch


                state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.cfg.device)
                with torch.no_grad():
                    last_val_A = self.agent_A.vnetwork(state_t).item()
                    z_last, _ = self.agent_A.pnetwork(state_t) 
                    dist_last = self.agent_A.pnetwork.get_dist(state_t)
                    z_last = dist_last.sample()
                    
                    combined_last = torch.cat([state_t, z_last], dim=1)
                    last_val_B = self.agent_B.vnetwork(combined_last).item()

                adv_A, ret_A = self.compute_gae(
                    common['rewards'], buf_A['values'], common['dones'], last_val_A
                )
                
                adv_B, ret_B = self.compute_gae(
                    common['rewards'], buf_B['values'], common['dones'], last_val_B
                )

                states_A = torch.tensor(np.array(buf_A['states']), dtype=torch.float32, device=self.cfg.device)
                actions_A = torch.tensor(np.array(buf_A['actions']), dtype=torch.float32, device=self.cfg.device)
                logp_A = torch.tensor(np.array(buf_A['log_probs']), dtype=torch.float32, device=self.cfg.device)
                
                states_B = torch.tensor(np.array(buf_B['states']), dtype=torch.float32, device=self.cfg.device)
                actions_B = torch.tensor(np.array(buf_B['actions']), dtype=torch.float32, device=self.cfg.device)
                logp_B = torch.tensor(np.array(buf_B['log_probs']), dtype=torch.float32, device=self.cfg.device)


                self.agent_A.update(states_A, actions_A, logp_A, ret_A, adv_A)
                
                self.agent_B.update(states_B, actions_B, logp_B, ret_B, adv_B)

                if total_steps % self.cfg.timesteps_per_batch == 0:
                    eval_reward = self.evaluate()
                    history['steps'].append(total_steps)
                    history['rewards'].append(eval_reward)
                    np.mean(history['rewards'][-100:])
                    print(f"[{env_name}] Step: {total_steps} | Eval Reward: {eval_reward:.2f}")
        
        except KeyboardInterrupt:
            print("Interrupted.")
        
        finally:
            self.env.close()
            self.eval_env.close()
        return history, np.array(z_ts), np.array(o_ts)

    def evaluate(self, n_rollouts=3):
        if hasattr(self.env, "obs_rms"):
            training_mean = self.env.obs_rms.mean
            training_var = self.env.obs_rms.var
            
            if hasattr(self.eval_env, "obs_rms"):
                self.eval_env.obs_rms.mean = training_mean
                self.eval_env.obs_rms.var = training_var
        
        rewards = []
        for _ in range(n_rollouts):
            state, _ = self.eval_env.reset()
            done = False
            total = 0.0
            while not done:
                state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.cfg.device)
                
                with torch.no_grad():
                    mu_a, _ = self.agent_A.pnetwork(state_t)
                    z_t = mu_a 
                    combined = torch.cat([state_t, z_t], dim=1)
                    
                    if self.is_continuous:
                        mu_b, _ = self.agent_B.pnetwork(combined)
                        action = mu_b.cpu().numpy()[0]
                    else:
                        logits = self.agent_B.pnetwork(combined)
                        action = torch.argmax(logits, dim=1).item()

                state, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                total += reward
            rewards.append(total)
        return np.mean(rewards)

    def plot_results(self, all_results):
        steps = [x[0] for x in all_results]
        rewards = [x[1] for x in all_results]
        plt.figure()
        plt.plot(steps, rewards)
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.title(f'Hierarchical PPO (A: Continuous, B: Discrete) on {self.cfg.env_name}')
        plt.savefig('training_curve_hierarchical.png')
        print("Saved plot to training_curve_hierarchical.png")

class GlobalTrainer_Single:
    def __init__(self, config: PPOConfig, env_name):
        self.cfg = config


        def make_env(is_training=True):
            if env_name in ['ALE/Frogger-v5']:
                env = gym.make(env_name, obs_type="ram" , frameskip=1, repeat_action_probability=0)
            else:
                env = gym.make(env_name)
            return env

        self.env = make_env(is_training=True)
        self.eval_env = make_env(is_training=False)
        self.obs_dim = self.env.observation_space.shape[0]

        self.is_continuous = isinstance(self.env.action_space, gym.spaces.Box)
        
        if self.is_continuous:
            act_dim = self.env.action_space.shape[0]
            self.agent_B = ContinuousPPOAgent(self.obs_dim, act_dim, self.cfg)
        else:
            act_dim = self.env.action_space.n 
            self.agent_B = DiscretePPOAgent(self.obs_dim, act_dim, self.cfg)
        


    def compute_gae(self, rewards, values, dones, last_value):
        N = len(rewards)
        advantages = np.zeros(N, dtype=np.float32)
        last_gae = 0.0
        
        for t in reversed(range(N)):
            if t == N - 1:
                next_value = last_value
                next_nonterminal = 0.0 if dones[t] else 1.0
            else:
                next_value = values[t + 1]
                next_nonterminal = 0.0 if dones[t + 1] else 1.0

            delta = rewards[t] + self.cfg.gamma * next_value * next_nonterminal - values[t]
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(values, dtype=np.float32)
        
        adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.cfg.device)
        ret_tensor = torch.tensor(returns, dtype=torch.float32, device=self.cfg.device)
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        return adv_tensor, ret_tensor

    def train(self,env_name):
        print(f"Training Hierarchical Agents on {env_name}")
        total_steps = 0
        history = {'steps': [], 'rewards': []}

        try:
            while total_steps < self.cfg.total_timesteps:
                buf_B = {'states': [], 'actions': [], 'log_probs': [], 'values': []}
                common = {'rewards': [], 'dones': []}
                
                state, _ = self.env.reset()
                
                for t in range(self.cfg.timesteps_per_batch):
                    state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.cfg.device)
                    
                    with torch.no_grad():
                        dist_b = self.agent_B.pnetwork.get_dist(state_t)
                        action_t = dist_b.sample()
                        val_b = self.agent_B.vnetwork(state_t)

                        if self.is_continuous:
                            logprob_b = dist_b.log_prob(action_t).sum(dim=-1)
                            env_action = action_t.cpu().numpy()[0] 
                            buf_action = action_t.cpu().numpy()[0]
                        else:
                            logprob_b = dist_b.log_prob(action_t)
                            env_action = action_t.item()
                            buf_action = action_t.item()
                    

                    next_state, reward, terminated, truncated, _ = self.env.step(env_action)
                    done = terminated or truncated

                    buf_B['states'].append(state)
                    buf_B['actions'].append(buf_action)
                    buf_B['log_probs'].append(logprob_b.item())
                    buf_B['values'].append(val_b.item())

                    common['rewards'].append(reward)
                    common['dones'].append(done)

                    state = next_state
                    if done:
                        state, _ = self.env.reset()

                total_steps += self.cfg.timesteps_per_batch

                state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.cfg.device)
                with torch.no_grad():
                    last_val_B = self.agent_B.vnetwork(state_t).item()


                adv_B, ret_B = self.compute_gae(
                    common['rewards'], buf_B['values'], common['dones'], last_val_B
                )
                

                states_B = torch.tensor(np.array(buf_B['states']), dtype=torch.float32, device=self.cfg.device)
                actions_B = torch.tensor(np.array(buf_B['actions']), dtype=torch.float32, device=self.cfg.device)
                logp_B = torch.tensor(np.array(buf_B['log_probs']), dtype=torch.float32, device=self.cfg.device)


                self.agent_B.update(states_B, actions_B, logp_B, ret_B, adv_B)


                if total_steps % self.cfg.timesteps_per_batch == 0:
                    #eval_reward = np.sum(common['rewards'])
                    eval_reward = self.evaluate()
                    history['steps'].append(total_steps)
                    history['rewards'].append(eval_reward)
                    np.mean(history['rewards'][-100:])
                    print(f"[{env_name}] Step: {total_steps} | Eval Reward: {eval_reward:.2f}")
        
        except KeyboardInterrupt:
            print("Interrupted.")
        
        finally:
            self.env.close()
            self.eval_env.close()
        return history

    def evaluate(self, n_rollouts=3):
        if hasattr(self.env, "obs_rms"):
            training_mean = self.env.obs_rms.mean
            training_var = self.env.obs_rms.var
            

            if hasattr(self.eval_env, "obs_rms"):
                self.eval_env.obs_rms.mean = training_mean
                self.eval_env.obs_rms.var = training_var
        

        rewards = []
        for _ in range(n_rollouts):
            state, _ = self.eval_env.reset()
            done = False
            total = 0.0
            while not done:
                state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.cfg.device)
                

                with torch.no_grad():

                    if self.is_continuous:

                        mu_b, _ = self.agent_B.pnetwork(state_t)
                        action = mu_b.cpu().numpy()[0]
                    else:

                        logits = self.agent_B.pnetwork(state_t)
                        action = torch.argmax(logits, dim=1).item()

                state, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                total += reward
            rewards.append(total)
        return np.mean(rewards)

    def plot_results(self, all_results):
        steps = [x[0] for x in all_results]
        rewards = [x[1] for x in all_results]
        plt.figure()
        plt.plot(steps, rewards)
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.title(f'Hierarchical PPO (A: Continuous, B: Discrete) on {self.cfg.env_name}')
        plt.savefig('training_curve_hierarchical.png')
        print("Saved plot to training_curve_hierarchical.png")

def plot_all_results(results_dict):
    n_envs = len(results_dict)
    cols = 3
    rows = math.ceil(n_envs / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    
    for idx, (env_name, data) in enumerate(results_dict.items()):
        ax = axes[idx]
        ax.plot(data['steps'], data['rewards'], label='PPO', color='b')
        ax.set_title(f"Environment: {env_name}")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Avg Reward")
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots if n_envs is not a multiple of cols
    for i in range(n_envs, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig('all_training_curves.png')
    print("\nCombined plotting complete. Saved to 'all_training_curves.png'")

def train_single_run(env_name: str, config: PPOConfig, n_runs):
    """Runs the training loop n_runs times and aggregates results."""
    print(f"\n{'='*60}")
    print(f"STARTING SINGLE-RUN TRAINING: {env_name}")
    print(f"Device: {config.device}")
    print(f"{'='*60}")
    
    all_runs_rewards = []
    
    for i in range(n_runs):
        
        print(f"  > Run {i+1}/{n_runs}...", end=" ", flush=True)
        
        # Train
        trainer = GlobalTrainer_Single(config,env_name)
        run_history = trainer.train(env_name)
        
        # Get rewards
        rewards_list = run_history['rewards']
        all_runs_rewards.append(rewards_list)
        
        print(f"Done. Final Reward: {rewards_list[-1]:.2f}")

    if not all_runs_rewards:
        return np.array([])

    min_len = min(len(r) for r in all_runs_rewards)
    all_runs_rewards = [r[:min_len] for r in all_runs_rewards]
    
    return np.array(all_runs_rewards)

def train_multi_run(env_name: str, config: PPOConfig, n_runs):
    """Runs the training loop n_runs times and aggregates results."""
    print(f"\n{'='*60}")
    print(f"STARTING MULTI-RUN TRAINING: {env_name}")
    print(f"Device: {config.device}")
    print(f"{'='*60}")
    
    all_runs_rewards = []
    
    for i in range(n_runs):
        print(f"  > Run {i+1}/{n_runs}...", end=" ", flush=True)
        
        # Train
        trainer = GlobalTrainer_Multi(config,env_name)
        run_history,z_ts,o_ts = trainer.train(env_name) # only getting last run, which is ideal
        
        # Get rewards
        rewards_list = run_history['rewards']
        all_runs_rewards.append(rewards_list)
        
        print(f"Done. Final Reward: {rewards_list[-1]:.2f}")

    if not all_runs_rewards:
        return np.array([])

    min_len = min(len(r) for r in all_runs_rewards)
    all_runs_rewards = [r[:min_len] for r in all_runs_rewards]
    
    return np.array(all_runs_rewards),z_ts,o_ts

def plot_combined_results(results_dict, steps_per_eval):
    """Saves the combined dashboard graph."""
    n_envs = len(results_dict)
    if n_envs == 0:
        print("No training results to plot.")
        return

    cols = 3
    rows = math.ceil(n_envs / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if n_envs == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    global_handles, global_labels = None, None
    
    for idx, (env_name, run_data) in enumerate(results_dict.items()):
        ax = axes[idx]
        
        mean_rewards = np.mean(run_data, axis=0)
        std_rewards = np.std(run_data, axis=0)
        steps = np.arange(1, len(mean_rewards) + 1) * steps_per_eval
        
        ax.plot(steps, mean_rewards, color='#0072B2', linewidth=2, label='Mean Reward')
        ax.fill_between(steps, mean_rewards - std_rewards, mean_rewards + std_rewards, color='#0072B2', alpha=0.2, label='Std Dev (±1σ)'
        )
        
        if idx == 0:
            global_handles, global_labels = ax.get_legend_handles_labels()

        max_mean = np.max(mean_rewards)
        final_mean = mean_rewards[-1]
        
        ax.set_title(f"{env_name}\nMax Mean: {max_mean:.1f} | Final Mean: {final_mean:.1f}")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Avg Reward")
        ax.grid(True, alpha=0.3)
    
    for i in range(n_envs, len(axes)):
        axes[i].axis('off')
        
    if global_handles:
        fig.legend(global_handles, global_labels, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=2, fontsize='medium', frameon=False
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join("graphs", 'all_envs_combined.png')
    plt.savefig(save_path)
    print(f"\nCombined plotting complete. Saved to '{save_path}'")

def plot_single_env_result(env_name, run_data, steps_per_eval):
    """Saves a standalone graph for a single environment."""
    
    # Calculate stats
    mean_rewards = np.mean(run_data, axis=0)
    std_rewards = np.std(run_data, axis=0)
    steps = np.arange(1, len(mean_rewards) + 1) * steps_per_eval
    
    # Metrics
    max_mean = np.max(mean_rewards)
    final_mean = mean_rewards[-1]

    plt.figure(figsize=(8, 6))
    
    plt.plot(steps, mean_rewards, color='#0072B2', linewidth=2, label='Mean Reward')
    plt.fill_between(steps, mean_rewards - std_rewards, mean_rewards + std_rewards, color='#0072B2', alpha=0.2, label='Std Dev (±1σ)'
    )
    
    plt.title(f"{env_name}\nMax Mean: {max_mean:.1f} | Final Mean: {final_mean:.1f}")
    plt.xlabel("Steps")
    plt.ylabel("Avg Reward")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    
    # Save with safe filename
    safe_name = env_name.replace("/", "_")
    filename = os.path.join("graphs", f"curve_single_{safe_name}.png")
    plt.savefig(filename)
    plt.close()
    
    print(f"Saved single plot to '{filename}'")

def store_data(env_name, data_z,data_o):
    safe_name = env_name.replace("/", "_")
    filename_z = os.path.join("extra_data", f"z_t_{safe_name}")
    filename_o = os.path.join("extra_data", f"o_t_{safe_name}")
    np.save(filename_z, data_z)
    np.save(filename_o, data_o)

if __name__ == '__main__':
    gym.register_envs(ale_py)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("graphs", exist_ok=True)
    os.makedirs("extra_data", exist_ok=True)
    cfg = PPOConfig()
    
    environments = [
        #"LunarLander-v3",
        "Ant-v5",
        #"HalfCheetah-v5",
        #"Hopper-v5",
        #"HumanoidStandup-v5",
        #"Humanoid-v5",
        #"InvertedDoublePendulum-v5",
        #"InvertedPendulum-v5",
        #"Pusher-v5",
        #"Reacher-v5",
        #"Swimmer-v5",
        #"Walker2d-v5"
    ]
    
    overall_results = {}
    N_RUNS = 3
    
    try:
        for env_name in environments:
            run_data,z_ts, o_ts = train_multi_run(env_name, cfg, n_runs=N_RUNS)
            overall_results[env_name] = run_data
            plot_single_env_result(env_name + "_multi", run_data, cfg.timesteps_per_batch)
            store_data(env_name,z_ts,o_ts)


            run_data = train_single_run(env_name, cfg, n_runs=N_RUNS)
            overall_results[env_name] = run_data
            plot_single_env_result(env_name + "_single", run_data, cfg.timesteps_per_batch)


                

    except KeyboardInterrupt:
        print("Exiting...")
        
    finally:
        if overall_results:
            print("\nAttempting to plot results collected so far...")

            plot_combined_results(overall_results, cfg.timesteps_per_batch)
