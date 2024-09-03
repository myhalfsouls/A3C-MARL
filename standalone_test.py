import torch
import random
from environment import environment_setup

# layout = "cramped_room"
# layout = "asymmetric_advantages"
# layout = "coordination_ring"
# layout = "forced_coordination"
layout = "counter_circuit_o_1order"

episode = 1499

params = {
    'seed': 0,
    'layer1_size': 512,
    'layer2_size': 128,
    'max_eps': 8000,
    'gamma': 0.99,
    'lambda': 1,
    'beta': 0.01,
    'alpha': 0.0001,
    'v_loss_ratio': 0.5,
    'num_workers': 10,
    'threshold': 64,
    'move_towards_serve_rew': 40,
    'update_freq': 500,
    'grad_clip': True,
    'grad_clip_value': 40,
    'device': 'cpu'
}

def potential(state):
    # define the x and y sizes for the specific layout
    y_size = 5
    x_size = 8
    phi = 1 - (abs(state[18]) + abs(state[19]))/(x_size + y_size)
    return phi

local_env = environment_setup(layout)
model_path = 'G:\\Dropbox (GaTech)\\CS7642_RLDM\\project3\\Curr_results\\test_' + str(episode) + '.pt'
localNN = torch.load(model_path)
localNN.eval()

rewards = []
rewards_shaped = []
rewards_phi = []

done = False
num_soup_made = 0

state = local_env.reset()
while not done:
    state0 = torch.tensor(state["both_agent_obs"][0], dtype=torch.float32, device=params['device'])
    state1 = torch.tensor(state["both_agent_obs"][1], dtype=torch.float32, device=params['device'])
    states = torch.tensor(state["both_agent_obs"][local_env.agent_idx], dtype=torch.float32, device=params['device'])
    a_dist0, a_dist1, _ = localNN(state0, state1, states)  # outputs 2 discrete distributions for actions (agent0, agent1), and value of state
    a0, a1 = random.choices(list(range(6)), a_dist0.tolist())[0], random.choices(list(range(6)), a_dist1.tolist())[0]  # get actions from distribution
    joint_action = [a0, a1]
    next_state, reward, done, info = local_env.step(joint_action)
    num_soup_made += int(reward/20)

    curr_held_obj = [state['overcooked_state'].players[0].held_object, state['overcooked_state'].players[1].held_object]
    next_held_obj = [next_state['overcooked_state'].players[0].held_object, next_state['overcooked_state'].players[1].held_object]
    phi_rewards = []
    for i in range(2):
        if info['shaped_r_by_agent'][i] == 5 and len(next_held_obj[i].ingredients) < 3: # agent picked up a soup from pot that is not complete
            info['shaped_r_by_agent'][i] = 2
            print(next_held_obj[i].ingredients)
        if curr_held_obj[i] is not None and curr_held_obj[i].name == 'soup': # and len(curr_held_obj[i].ingredients) == 3: # player i has a completed soup in hand
            phi_reward = (params['gamma'] * potential(next_state["both_agent_obs"][i]) - potential(state["both_agent_obs"][i])) * params['move_towards_serve_rew']
            phi_rewards.append(phi_reward)
    full_reward = reward + sum(info['shaped_r_by_agent']) + sum(phi_rewards)
    # end this abominable creation

    # full_reward = reward + sum(info['shaped_r_by_agent'])
    rewards.append(full_reward)
    rewards_shaped.append(sum(info['shaped_r_by_agent']))
    rewards_phi.append(sum(phi_rewards))
    state = next_state


print(rewards)
print(rewards_shaped)
print(rewards_phi)
print(sum(rewards))