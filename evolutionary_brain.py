import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ArchitecturallyCorrectRNN(nn.Module):
    def __init__(self, n=1000, state_size=105, output_size=8, sparsity=1):
        super().__init__()
        self.n, self.state_size, self.output_size = n, state_size, output_size
        self.shape = (n + output_size, n + state_size) # 
        
        # Quadrant-Specific Initialization 
        nnz_wa = int(n * n * sparsity)
        k = max(2, nnz_wa // n)
        wa_rows = torch.arange(n).repeat_interleave(k)
        wa_cols = (wa_rows + torch.arange(1, k + 1).repeat(n)) % n # Small-world base
        
        # Interface connections [cite: 9, 13]
        ws_rows = torch.randint(0, n, (int(n * state_size * sparsity),))
        ws_cols = torch.randint(n, n + state_size, (len(ws_rows),))
        ma_rows = torch.randint(n, n + output_size, (int(output_size * n * sparsity),))
        ma_cols = torch.randint(0, n, (len(ma_rows),))
        ms_rows = torch.randint(n, n + output_size, (int(output_size * state_size * sparsity),))
        ms_cols = torch.randint(n, n + state_size, (len(ms_rows),))
        
        self.register_buffer('indices', torch.stack([
            torch.cat([wa_rows, ws_rows, ma_rows, ms_rows]),
            torch.cat([wa_cols, ws_cols, ma_cols, ms_cols])
        ]))
        # Start with a random 'seed' brain [cite: 12, 16]
        self.register_buffer('values', torch.randn(self.indices.size(1)) * (0.01 / np.sqrt(n)))

    def forward(self, a_t, s_t, custom_values=None):
        v = custom_values if custom_values is not None else self.values
        x = torch.cat([a_t, s_t], dim=-1).t()
        weights = torch.sparse_coo_tensor(self.indices, v, self.shape, device=device)
        z = torch.sparse.mm(weights, x).t()
        activated = torch.sigmoid(z) # [cite: 1]
        return activated[:, :self.n], (activated[:, self.n:] * 2) - 1

def evaluate(model, env, vals):
    obs, _ = env.reset()
    a_t = torch.zeros(1, model.n, device=device)
    total_reward = 0
    for _ in range(1000): # Max steps
        with torch.no_grad():
            a_t, action = model(a_t, torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0), vals)
        obs, reward, term, trunc, _ = env.step(action[0].cpu().numpy())
        total_reward += reward
        if term or trunc: break
    return total_reward

def train_elite_strategy():
    env = gym.make("Ant-v5")
    model = ArchitecturallyCorrectRNN().to(device)
    
    # Elitism Setup
    elite_vals = copy.deepcopy(model.values)
    best_reward = evaluate(model, env, elite_vals)
    
    sigma = 0.003  # Mutation strength
    pop_size = 50 # Number of mutations to check per "Elite"
    
    print(f"Initial Best Reward: {best_reward:.2f}")

    for gen in range(1000):
        # 1. Mutate: Sample noise around the current Elite
        found_better = False
        for i in range(pop_size):
            mutation = elite_vals + torch.randn_like(elite_vals) * sigma
            reward = evaluate(model, env, mutation)
            print(f"Gen {gen} | Mut {i} | Reward: {reward:.2f}")
            
            # 2. Selection: Only keep if strictly better (Hill Climbing)
            if reward > best_reward:
                best_reward = reward
                elite_vals = mutation
                found_better = True
                print(f"Gen {gen} | New Elite! Reward: {best_reward:.2f}")
        
        # 3. Success-Based Step: If we aren't finding better ants, increase search radius
        if not found_better:
            sigma *= .95
        else:
            sigma *= 1.0 # Tighten search around the new champion
            sigma = max(sigma, 0.003)

if __name__ == "__main__":
    train_elite_strategy()