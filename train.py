import gym
import os
import numpy as np
from bpo_env2 import BPOEnv

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib import MaskablePPO
from stable_baselines3.common.evaluation import evaluate_policy

from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.monitor import Monitor

from plotting_helpers import plot_results
import matplotlib.pyplot as plt

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.logger import configure




from stable_baselines3.common.vec_env import SubprocVecEnv

class CustomPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 128, 128],
                                                          vf=[128, 128, 128])])

if __name__ == '__main__':
    #env = InvalidActionEnvDiscrete(dim = 3, n_invalid_actions=1)

    #if true, load model for a new round of training
    load_model = False
    model_name = "ppo_masked_test"

    # Create log dir
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)

    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    # Create and wrap the environment
    env = Monitor(BPOEnv(instance_file="./gym_bpo/envs/BPI Challenge 2017 - instance.pickle"))  # Initialize env
    #vec_env = BPOEnv(instance_file="./gym_bpo/envs/BPI Challenge 2017 - instance.pickle")
    #env = SubprocVecEnv([make_env(vec_env, i) for i in range(2)])
    #env = make_vec_env(vec_env, 2)

    # MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
    # with ActionMasker. If the wrapper is detected, the masks are automatically
    # retrieved and used when learning. Note that MaskablePPO does not accept
    # a new action_mask_fn kwarg, as it did in an earlier draft.
    obs = env.reset()

    if load_model:
        model = MaskablePPO.load(model_name, env, learning_rate=0.0001, n_steps = 200, gamma=0.99, verbose=1)
    else:
        model = MaskablePPO(CustomPolicy, env, learning_rate=0.0001, n_steps = 200, gamma=0.99, verbose=1)


    model.set_logger(new_logger)

    timesteps = 500000

    model.learn(int(timesteps), eval_log_path=log_dir)

    #plot_results(log_dir)
    #plt.show()


    model.save(model_name)

    print(f"Model saved with the name: {model_name}")


    # Evaluate the policy
    #mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)

    #print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")