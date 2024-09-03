import torch
import random
from environment import environment_setup
from model_build import NN
from collections import deque
import statistics

# training with A3C algorithm, centralized critic
def A3C_train_sumR(layout, rank, params, globalNN, globalOpt):
    torch.manual_seed(params['seed'] + rank)  # set seed with rank for multiprocessing: https://github.com/pytorch/examples/blob/main/mnist_hogwild/train.py
    local_env = environment_setup(layout)
    state_space = local_env.observation_space.shape[0]
    action_space = local_env.action_space.n
    localNN = NN(state_space, params['layer1_size'], params['layer2_size'], action_space).to(params['device'])  # create a network specific to worker

    soup_ep = []
    soup_avg100 = []
    avg100 = deque([], maxlen=100)
    reward_ep = []
    reward_avg100 = []
    ravg100 = deque([], maxlen=100)

    infos = {}

    # iterate through episodes
    for ep in range(params['max_eps']):
        localNN.load_state_dict(globalNN.state_dict())  # sync worker network with global network
        localNN.eval()  # local network will never be in training mode, since it's weights do not get updated. It just draws weight values from global network

        state = local_env.reset()

        values = []
        entropies0 = []
        entropies1 = []
        log_a_probs0 = []
        log_a_probs1 = []
        rewards = []

        done = False
        num_soup_made = 0

        outstanding_soup = 0

        # iterate through timesteps until 400 timesteps have elapsed
        while not done:
            state0 = torch.tensor(state["both_agent_obs"][0], dtype=torch.float32, device=params['device'])
            state1 = torch.tensor(state["both_agent_obs"][1], dtype=torch.float32, device=params['device'])
            states = torch.tensor(state["both_agent_obs"][local_env.agent_idx], dtype=torch.float32, device=params['device'])
            a_dist0, a_dist1, value = localNN(state0, state1, states)  # outputs 2 discrete distributions for actions (agent0, agent1), and value of state
            values.append(value)
            log_a_dist0, log_a_dist1 = localNN.log_a_dists(state0, state1) # calculate natural log of distribution
            entropy0, entropy1 = -torch.sum(log_a_dist0 * a_dist0), -torch.sum(log_a_dist1 * a_dist1)  # calculate entropy
            entropies0.append(entropy0)
            entropies1.append(entropy1)
            a0, a1 = random.choices(list(range(6)), a_dist0.tolist())[0], random.choices(list(range(6)), a_dist1.tolist())[0]  # get actions from distribution
            log_a_prob0, log_a_prob1 = log_a_dist0.gather(0, torch.tensor(a0, device=params['device'])), log_a_dist1.gather(0, torch.tensor(a1, device=params['device']))  # choose the log probability value corresponding to the action
            log_a_probs0.append(log_a_prob0)
            log_a_probs1.append(log_a_prob1)
            joint_action = [a0, a1]
            next_state, reward, done, info = local_env.step(joint_action)
            num_soup_made += int(reward/20)
            outstanding_soup -= int(reward/20)

            curr_held_obj = [state['overcooked_state'].players[local_env.agent_idx].held_object, state['overcooked_state'].players[1-local_env.agent_idx].held_object]
            next_held_obj = [next_state['overcooked_state'].players[local_env.agent_idx].held_object, next_state['overcooked_state'].players[1-local_env.agent_idx].held_object]
            if info['shaped_r_by_agent'][0] == 5:
                if next_held_obj[local_env.agent_idx] is not None and next_held_obj[local_env.agent_idx].name == 'soup' and len(next_held_obj[local_env.agent_idx].ingredients) == 3:  # picked up a 3 onion soup
                    outstanding_soup += 1
            if info['shaped_r_by_agent'][1] == 5:
                if next_held_obj[1-local_env.agent_idx] is not None and next_held_obj[1-local_env.agent_idx].name == 'soup' and len(next_held_obj[1-local_env.agent_idx].ingredients) == 3:  # picked up a 3 onion soup
                    outstanding_soup += 1

            if outstanding_soup >= 3:
                phi_rewards = []
                for i in range(2):
                    if curr_held_obj[i] is not None and curr_held_obj[i].name == 'soup' and len(curr_held_obj[i].ingredients) == 3: # player i has a completed soup in hand
                        if next_held_obj[i] is not None and next_held_obj[i].name == 'soup' and len(next_held_obj[i].ingredients) == 3:  # player i did not drop soup
                            phi_reward = (params['gamma'] * potential(next_state["both_agent_obs"][i]) - potential(state["both_agent_obs"][i])) * params['phi_weight']
                            phi_rewards.append(phi_reward)
                full_reward = reward + sum(info['shaped_r_by_agent']) + sum(phi_rewards)
            else:
                full_reward = reward + sum(info['shaped_r_by_agent'])
            rewards.append(full_reward)
            state = next_state

        state0 = torch.tensor(state["both_agent_obs"][0], dtype=torch.float32, device=params['device'])
        state1 = torch.tensor(state["both_agent_obs"][1], dtype=torch.float32, device=params['device'])
        states = torch.tensor(state["both_agent_obs"][local_env.agent_idx], dtype=torch.float32, device=params['device'])
        _, _, value_terminal = localNN(state0, state1, states)  # get the value of the state at termination
        R = value_terminal.detach()  # r_T
        values.append(value_terminal)
        MC_R = torch.zeros(1, 1)
        p0_loss = 0
        p1_loss = 0
        v_loss = 0

        for i in reversed(range(len(rewards))):
            R = rewards[i] + params['gamma'] * R  # R <-- r_t + gamma*R for t in {T-1, T-2, ......, 0}
            Adv = R - values[i]  # calculate advantage for every time step in the episode
            v_loss += 0.5 * (Adv).pow(2)  # accumulate value loss as (advantage_t)^2, equivalent to accumulating gradients
            MC_R = params['gamma'] * MC_R + rewards[i] + params['gamma'] * values[i + 1] - values[i]  # Generalized advantage estimation
            p0_loss = p0_loss - log_a_probs0[i] * MC_R.detach() - params['beta'] * entropies0[i]  # accumulate policy loss
            p1_loss = p1_loss - log_a_probs1[i] * MC_R.detach() - params['beta'] * entropies1[i]
        total_loss = 0.5 * v_loss + p0_loss + p1_loss

        globalOpt.zero_grad()
        total_loss.backward()
        if params['grad_clip']:
            torch.nn.utils.clip_grad_value_(localNN.parameters(), params['grad_clip_value'])
        for global_param, local_param in zip(globalNN.parameters(), localNN.parameters()):
            global_param._grad = local_param.grad
        globalOpt.step()

        soup_ep.append(num_soup_made)
        avg100.append(num_soup_made)
        soup_avg100.append(statistics.mean(avg100))
        reward_ep.append(sum(rewards))
        ravg100.append(sum(rewards))
        reward_avg100.append(statistics.mean(ravg100))

        print('Worker # {}: Ep {}, config {}, # soups made {}, total loss {:.3f}, total reward {}'.format(rank, ep, local_env.agent_idx, num_soup_made, total_loss.item(), sum(rewards)))

        if rank == 0 and (ep+1) % params['update_freq'] == 0:
            torch.save(globalNN, 'test_' + str(ep) + '.pt')

def potential(state):
    phi = -(abs(state[18]) + abs(state[19]))
    return phi