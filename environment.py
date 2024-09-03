from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
import gym

def environment_setup(layout):
    layout = layout

    # Reward shaping is disabled by default.  This data structure may be used for
    # reward shaping.  You can, of course, do your own reward shaping in lieu of, or
    # in addition to, using this structure.
    reward_shaping = {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 2,
        "SOUP_PICKUP_REWARD": 5
    }

    # additional_r_shaping = {
    #     'useful_onion_pickup': 1,
    #     'useful_onion_drop': 3,
    #     'potting_onion': 3,
    #     'useful_dish_pickup': 3,
    #     'useful_dish_drop': 1,
    #     'soup_pickup': 5,
    # }

    # Length of Episodes.  Do not modify for your submission!
    # Modification will result in a grading penalty!
    horizon = 400

    # Build the environment.  Do not modify!
    mdp = OvercookedGridworld.from_layout_name(layout, rew_shaping_params=reward_shaping)
    base_env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
    env = gym.make("Overcooked-v0", base_env=base_env, featurize_fn=base_env.featurize_state_mdp, disable_env_checker = True)

    return env