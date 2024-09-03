import torch.multiprocessing as mp
from environment import environment_setup
import model_build
import model_train

# Hyperparameters and settings
params = {
    'seed': 0,
    'layer1_size': 512,
    'layer2_size': 128,
    'max_eps': 5000,
    'gamma': 0.99,
    'beta': 0.01,
    'alpha': 0.0001,
    'num_workers': 10,
    'update_freq': 500,
    'grad_clip': True,
    'grad_clip_value': 40,
    'phi_weight': 3,
    'device': 'cpu'
}

# Swap between the 5 layouts here:
layout = "cramped_room"
# layout = "asymmetric_advantages"
# layout = "coordination_ring"
# layout = "forced_coordination"
# layout = "counter_circuit_o_1order"

params['layout'] = layout

env = environment_setup(params['layout'])
state_space = env.observation_space.shape[0]
action_space = env.action_space.n

globalNN = model_build.NN(state_space, params['layer1_size'], params['layer2_size'], action_space).to(params['device'])
target = model_train.A3C_train_sumR

globalNN.share_memory()
globalOpt = model_build.SharedAdam(globalNN.parameters(), lr=params['alpha'])
globalNN.train()

if __name__ == '__main__':
    workers = []
    for rank in range(params['num_workers']):
        worker = mp.Process(target=target, args=(layout, rank, params, globalNN, globalOpt))
        worker.start()
        workers.append(worker)
    for worker in workers:
        worker.join()
