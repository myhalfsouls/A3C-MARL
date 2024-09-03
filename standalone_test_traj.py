import torch
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.agents.agent import NNPolicy, AgentFromPolicy, AgentPair
from PIL import Image
import os

episode = 4999

### Environment setup ###

# Swap between the 5 layouts here:
# layout = "cramped_room"
# layout = "asymmetric_advantages"
# layout = "coordination_ring"
layout = "forced_coordination"
# layout = "counter_circuit_o_1order"

# Reward shaping is disabled by default.  This data structure may be used for
# reward shaping.  You can, of course, do your own reward shaping in lieu of, or
# in addition to, using this structure.
reward_shaping = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 2,
    "SOUP_PICKUP_REWARD": 5
}

# Length of Episodes.  Do not modify for your submission!
# Modification will result in a grading penalty!
horizon = 400

# Build the environment.  Do not modify!
mdp = OvercookedGridworld.from_layout_name(layout, rew_shaping_params=reward_shaping)
base_env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)

### Evaluate your agent ###

# This is where you would rollout episodes with your trained agent.
# The below code is a particular way to rollout episodes in a format
# compatible with a state visualizer, if you'd like to visualize what your
# agents are doing during episodes.  Visualization is in the next cell.

model_path = 'G:\\Dropbox (GaTech)\\CS7642_RLDM\\project3\\layout4\\independentSumR\\test_' + str(episode) + '.pt'

trainedNN = torch.load(model_path)
trainedNN.eval()

class StudentPolicy(NNPolicy):
    """ Generate policy """
    def __init__(self):
        super(StudentPolicy, self).__init__()

    def state_policy(self, state, agent_index):
        """
        This method should be used to generate the poiicy vector corresponding to
        the state and agent_index provided as input.  If you're using a neural
        network-based solution, the specifics depend on the algorithm you are using.
        Below are two commented examples, the first for a policy gradient algorithm
        and the second for a value-based algorithm.  In policy gradient algorithms,
        the neural networks output a policy directly.  In value-based algorithms,
        the policy must be derived from the Q value outputs of the networks.  The
        uncommented code below is a placeholder that generates a random policy.
        """
        featurized_state = base_env.featurize_state_mdp(state)
        input_state0 = torch.tensor(featurized_state[0], dtype=torch.float32).unsqueeze(0)
        input_state1 = torch.tensor(featurized_state[1], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
          if agent_index == 0:
              a_dist, _, _, _ = trainedNN(input_state0, input_state1)
              a_dist = a_dist[0].numpy()
          else:
              _, a_dist, _, _ = trainedNN(input_state0, input_state1)
              a_dist = a_dist[0].numpy()

        return a_dist

    def multi_state_policy(self, states, agent_indices):
        """ Generate a policy for a list of states and agent indices """
        return [self.state_policy(state, agent_index) for state, agent_index in zip(states, agent_indices)]


class StudentAgent(AgentFromPolicy):
    """Create an agent using the policy created by the class above"""
    def __init__(self, policy):
        super(StudentAgent, self).__init__(policy)


# Instantiate the policies for both agents
policy0 = StudentPolicy()
policy1 = StudentPolicy()

# Instantiate both agents
agent0 = StudentAgent(policy0)
agent1 = StudentAgent(policy1)
agent_pair = AgentPair(agent0, agent1)

# Generate an episode
ae = AgentEvaluator.from_layout_name({"layout_name": layout}, {"horizon": horizon})
trajs = ae.evaluate_agent_pair(agent_pair, num_games=1)
print("\nlen(trajs):", len(trajs))


# Modify as appropriate
img_dir = 'G:\\Dropbox (GaTech)\\CS7642_RLDM\\project3\\Curr_images\\' + str(episode) + '\\'
ipython_display = True
gif_path = 'G:\\Dropbox (GaTech)\\CS7642_RLDM\\project3\\Curr_images\\' + str(episode) + '\\result.gif'

# Do not modify -- uncomment for GIF generation
StateVisualizer().display_rendered_trajectory(trajs, img_directory_path=img_dir, ipython_display=ipython_display)

# The following code generates the gif
img_list = [f for f in os.listdir(img_dir) if f.endswith('.png')]
img_list.sort(key=lambda x: os.path.getmtime(os.path.join(img_dir, x)))
images = [Image.open(img_dir + img).convert('RGBA') for img in img_list]
images[0].save(gif_path, save_all=True, append_images=images[1:], optimize=False, duration=250, loop=0)