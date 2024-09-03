import torch
import random
from environment import environment_setup

# Swap between the 5 layouts here:
layout = "cramped_room"
# layout = "asymmetric_advantages"
# layout = "coordination_ring"
# layout = "forced_coordination"
# layout = "counter_circuit_o_1order"

local_env = environment_setup(layout)
model_path = 'test_499.pt'
trained_model = torch.load(model_path)
trained_model.eval()

soups = []
configs = []

for ep in range(100):
    done = False
    num_soup_made = 0
    state = local_env.reset()
    while not done:
        state0 = torch.tensor(state["both_agent_obs"][0], dtype=torch.float32, device='cpu')
        state1 = torch.tensor(state["both_agent_obs"][1], dtype=torch.float32, device='cpu')
        states = torch.tensor(state["both_agent_obs"][0], dtype=torch.float32, device='cpu')
        a_dist0, a_dist1, _ = trained_model(state0, state1, states)  # outputs 2 discrete distributions for actions (agent0, agent1), and value of state
        a0, a1 = random.choices(list(range(6)), a_dist0.tolist())[0], random.choices(list(range(6)), a_dist1.tolist())[0]  # get actions from distribution
        joint_action = [a0, a1]
        next_state, reward, done, info = local_env.step(joint_action)
        num_soup_made += int(reward/20)
        state = next_state
    soups.append(num_soup_made)
    configs.append(local_env.agent_idx)
    print('EP {}: {} soups made'.format(ep, num_soup_made))

print(configs)
print(soups)