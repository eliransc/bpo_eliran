from collections import deque
from subprocess import call
import gym
import os
import numpy as np
from bpo_env2 import BPOEnv

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy, MaskableMultiInputActorCriticPolicy
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure

from gym.wrappers import normalize
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


class CustomPolicy(MaskableActorCriticPolicy):
    """ 
    Define NN architecture
    """
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 128, 128],
                                                          vf=[128, 128, 128])])

class LogTraining(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(LogTraining, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')

        self.episode_reward = 0        
        self.episode_rewards = []
        self.episode = []

        self.timestep = []
        self.tmp_scores = deque(maxlen=check_freq)
        self.avg_scores = []

        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        self.tmp_scores.append(reward)
        if self.n_calls % self.check_freq == 0:
            self.timestep.append(self.n_calls)
            self.avg_scores.append(sum(self.tmp_scores))
            print(f'Timestep: {self.n_calls}. Reward during previous {self.check_freq} timesteps: {sum(self.tmp_scores)}')

        self.episode_reward += reward
        return True

    # def _on_rollout_start(self) -> None:
    #     self.episode_reward = 0

    # def _on_rollout_end(self) -> None:
    #     self.episode.append(len(self.episode))
    #     self.episode_rewards.append(self.episode_reward)
    #     print(self.episode_reward)

if __name__ == '__main__':
    #if true, load model for a new round of training
    
    running_time = 365*24
    num_cpu = 2
    load_model = False
    model_name = "ppo_masked_long_train_time"
    
    # Create log dir
    log_dir = "./tmp/"
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    if num_cpu > 1:
      env = SubprocVecEnv([lambda: Monitor(BPOEnv(instance_file="./gym_bpo/envs/BPI Challenge 2017 - instance.pickle", running_time = running_time)) for _ in range(num_cpu)] )
    else:
      env = BPOEnv(instance_file="./gym_bpo/envs/BPI Challenge 2017 - instance.pickle", running_time = running_time)  # Initialize env
      env = Monitor(env, log_dir)  
      
    #env = normalize.NormalizeReward(env) #rewards normalization
    #env = normalize.NormalizeObservation(env) #rewards normalization

    # Create the model
    if load_model:
        model = MaskablePPO.load(model_name, env, batch_size=3000, gae_lambda=0.99, vf_coef=0.95, ent_coef=0.01, learning_rate=0.0003, n_steps = 3000, gamma=0.9, verbose=1, tensorboard_log="./tmp/")
    else:
        model = MaskablePPO(MaskableActorCriticPolicy, env, ent_coef=0, learning_rate=0.0003, n_steps = 3000, gamma=0.3, verbose=1, tensorboard_log="./tmp/")

    #Logging to tensorboard. To access tensorboard, open a bash terminal in the projects directory, activate the environment (where tensorflow should be installed) and run the command in the following line
    # tensorboard --logdir ./tmp/
    # then, in a browser page, access localhost:6006 to see the board
    model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

    # Create the callbacks list
    log_callback = LogTraining(check_freq=300, log_dir=log_dir)

    checkpoint_callback = CheckpointCallback(
      save_freq=50000,
      save_path="./",
      name_prefix="ppo_masked_long_train_time",
    )

    callback = CallbackList([log_callback, checkpoint_callback])

    # Train the agent
    time_steps = 5000

    model.learn(total_timesteps=int(time_steps), callback=callback, tb_log_name="first_run", reset_num_timesteps=False)


    # For episode rewards, use env.get_episode_rewards()
    # env.get_episode_times() returns the wall clock time in seconds of each episode (since start)
    # env.rewards returns a list of ALL rewards. Of current episode?
    # env.episode_lengths returns the number of timesteps per episode
    if num_cpu==1:
        print(env.get_episode_rewards())
        print(env.get_episode_times())


    model.save(model_name)

