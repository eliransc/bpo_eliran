from re import S
import gym
from gym import spaces, Env
import random
import numpy as np
from typing import List

from simulator import Simulator

class BPOEnv(Env):
    def __init__(self, instance_file="./gym_bpo/envs/BPI Challenge 2017 - instance.pickle", running_time = 365*24) -> None:
        super().__init__()
        self.num_envs = 1
        self.instance_file = instance_file
        self.running_time = running_time

        self.simulator = Simulator(running_time=self.running_time, report=False, instance_file=self.instance_file, planner = None)
        
        #define lows and highs for different sections of the input
        lows = np.array([0 for x in range(len(self.simulator.input))])
        
        highs = np.array([1 for x in range(len(self.simulator.resources))] + [np.finfo(np.float64).max for x in range(len(self.simulator.resources))] + [len(self.simulator.task_types) for x in range(len(self.simulator.resources))] + [np.finfo(np.float64).max for x in range(len(self.simulator.task_types))])

        #first len(resources): ones for available, zero for busy/away; second len(resources): time passed in current task for each resource (zero if free or away); third len(resources): current task a resource is doing (zero if free or away); len(task types): number of each type of task in waiting queue
        self.observation_space = gym.spaces.Box(low=np.float32(lows), high=np.float32(highs))#Box(low=0, high=np.finfo(np.int64).max, shape=shape(self.input), dtype=np.int64) #observation space is the concatentenation of resource pools
        self.action_space = spaces.Discrete(len(self.simulator.output)) #action space is the cartesian product of tasks and resources in their resource pool
        
        self.observation_space = spaces.Box(low=lows, 
                                            high=highs,
                                            shape=(len(self.simulator.input),), dtype=np.float64) #observation space is the cartesian product of resources and tasks

        # spaces.Discrete returns a number between 0 and len(self.simulator.output)
        self.action_space = spaces.Discrete(len(self.simulator.output)) #action space is the cartesian product of tasks and resources in their resource pool


        self.test_zero = 0
        self.test_more = []
        while (sum(self.define_action_masks()) <= 1):
            self.simulator.run() # Run the simulator to get to the first decision


    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
       
        # 1 Process action
        # 2 Do the timestep
        # 3 Return reward
        # Assign one resources per iteration. If possible, another is assigned in next step without advancing simulator
        assignment = self.simulator.output[action]
        #print(assignment)
        if assignment != 'postpone':
            #print('stuck 1', assignment, self.simulator.now)
            self.simulator.schedule_resources([assignment])
            # While assignment not possible and simulator not finished (postpone always possible)
            while (sum(self.define_action_masks()) <= 1) and (self.simulator.status != 'FINISHED'):
                #print('ASSIGNED', self.simulator.now)
                self.simulator.run() # breaks each time at resource assignment, continues if no assignment possible
        else: # Postpone action

            #print('Postpone')

            unassigned_tasks = [self.simulator.unassigned_tasks[task].id for task in self.simulator.unassigned_tasks]

            available_resources = [resource for resource in self.simulator.available_resources]


            while (self.simulator.status != 'FINISHED') and (sum(self.define_action_masks()) <= 1) or (unassigned_tasks == [self.simulator.unassigned_tasks[task].id for task in self.simulator.unassigned_tasks] and \

                    available_resources == [resource for resource in self.simulator.available_resources]):

                self.simulator.run()
                

        # Simulation is finished, return current reward (with penalties)
        if self.simulator.status == 'FINISHED':
            print('FINAL REWARD', self.simulator.current_reward)
            return self.simulator.get_state(), self.simulator.current_reward, True, {}

        reward = self.simulator.current_reward
        self.simulator.current_reward = 0

        return self.simulator.get_state(), reward, False, {}


    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.

        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.

        Returns:
            observation (object): the initial observation.
        """

        print("-------Resetting environment-------")

        self.simulator = Simulator(running_time=self.running_time, report=False, instance_file=self.instance_file, planner = None)
        while (sum(self.define_action_masks()) <= 1):
            self.simulator.run() # Run the simulator to get to the first decision       
        
        #self.finished = False
        return self.simulator.get_state()


    def render(self, mode='human', close=False):
        print(f"Average reward: {self.average_cycle_time}")

    # define mask based on current environment state (only the 3 vectors that are also known at inference time!)
    def define_action_masks(self) -> List[bool]:
        state = self.simulator.get_state()

        mask = [0 for _ in range(len(self.simulator.output))]
        for i in range(len(self.simulator.resources)*3, len(self.simulator.resources)*3 + len(self.simulator.task_types)):
            if state[i] > 0: # Check for available tasks
                task = self.simulator.input[i] # Get task string
                for resource_index in self.simulator.resource_pools_indexes[task]: # type(resource_pools_indexes) = dict 
                    if state[resource_index] > 0: # If we have a task, check which resources can perform that task
                        mask[self.simulator.output.index((task, self.simulator.input[resource_index]))] = 1
        mask[-1] = 1 # Set postpone action to 1
        invalid_actions = list(map(bool, mask))
        return invalid_actions

    def action_masks(self) -> List[bool]:
        return self.define_action_masks()

    """
    TRAINING
    Needed:
        >Functions:
            -Simulator step function: continues until plan is called
            -Check output function
            -Get state function
            -Action function
                *If multiple actions necessary -> better to invoke step function multiple times
                and pass the assignments to the simulator once
            -Reward function
        >Adjustments:
            -Sample interarrivals during training (no fixed file)

    Optional:
        -Use Env.close() -> disposes all garbage

    INFERENCE
    """