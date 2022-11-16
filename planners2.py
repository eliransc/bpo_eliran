from abc import ABC, abstractmethod
from sb3_contrib.ppo_mask import MaskablePPO, MlpPolicy
from bpo_env import BPOEnv
import numpy as np
import json
import pandas as pd
from itertools import permutations, combinations
import time

class Planner(ABC):
    """Abstract class that all planners must implement."""

    @abstractmethod
    def plan(self, available_resources, unassigned_tasks, resource_pool):
        """
        Assign tasks to resources from the simulation environment.

        :param environment: a :class:`.Simulator`
        :return: [(task, resource, moment)], where
            task is an instance of :class:`.Task`,
            resource is one of :attr:`.Problem.resources`, and
            moment is a number representing the moment in simulation time
            at which the resource must be assigned to the task (typically, this can also be :attr:`.Simulator.now`).
        """
        raise NotImplementedError

# DRL based assignment
class PPOPlanner(Planner):
    """A :class:`.Planner` that assigns tasks to resources following policy dictated by (pretrained) DRL algorithm."""

    def __init__(self, model_name="ppo_masked_long_train") -> None:
        self.model = MaskablePPO.load(model_name)
        self.resources = None
        self.task_types = None
        self.inputs = None
        self.output = []
        self.resource_pools_indexes = {}
        self.reward_interval = 3

        self.simulator = None

    def getState(self, available_resources, unassigned_tasks, busy_resources):
        
        #av_resources_ones = list(np.where(np.isin(np.asarray(self.resources), self.available_resources), 1, 0))
        av_resources_ones = [1 if x in available_resources else 0 for x in self.resources]
        
        #get current time in task for each resource (x[1] is the processing start time)
        busy_resources_times = [self.simulator.now - busy_resources[x][1] if x in busy_resources else 0 for x in self.resources]
        
        #get current task type for each busy resource
        busy_resources_tasks = [self.task_types.index(busy_resources[x][0].task_type) + 1 if x in busy_resources else 0 for x in self.resources]

        #task_types_num = [np.count_nonzero(el in [o.task_type for o in self.unassigned_tasks.values()]) for el in self.task_types]
        task_types_num =  [np.sum(el in [o.task_type for o in unassigned_tasks]) for el in self.task_types]
        
        return av_resources_ones + busy_resources_times + busy_resources_tasks + task_types_num

    #pass the simulator for bidirectional communication
    def linkSimulator(self, simulator):
        self.simulator = simulator


    def getActionMasks(self, state):
        mask = [0 for _ in range(len(self.output))]
        for i in range(len(self.resources)*3, len(self.inputs)):
            if state[i] > 0: # Check for available tasks
                task = self.inputs[i] # Get task string
                for resource_index in self.resource_pools_indexes[task]:
                    if state[resource_index] > 0:
                        mask[self.output.index((task, self.inputs[resource_index]))] = 1
        
        mask[-1] = 1 #"do nothing" is always allowed                  
        self.invalid_actions = list(map(bool,mask))
        return self.invalid_actions

    def take_action(self, action):   
        return self.output[action]

    def plan(self, available_resources, unassigned_tasks, resource_pool):
        #notice these values could be calculated only once at the instantiation but we kept it like this to match the required code structure
        if (self.resources == None) and (self.task_types == None):
            self.resources = sorted(list(set(np.hstack(list(resource_pool.values()))))) #all the resources in the problem (should be 145 elements)
            self.task_types = sorted(list(resource_pool.keys()))
            self.inputs = self.resources + self.resources + self.resources + self.task_types #input example

            
            for task, value in resource_pool.items():
                indexes = []
                for resource in value:
                    indexes.append(self.resources.index(resource)) # Finds and appends the index of a resource
                    self.output.append((task, resource))
        
                self.resource_pools_indexes[task] = indexes
            self.output.append('postpone')
        
        assignments = []

        available_resources = available_resources.copy() #is this useful?
        unassigned_tasks = unassigned_tasks.copy() # I dont know but let's do it
        busy_resources = self.simulator.busy_resources.copy()


        while (sum(self.getActionMasks(self.getState(available_resources, unassigned_tasks, busy_resources)))) > 1:
            obs = self.getState(available_resources, unassigned_tasks, busy_resources)
            action_masks = self.getActionMasks(obs)
            action, _states = self.model.predict(obs, action_masks=action_masks)
            if self.output[action] == 'postpone':
                #print("POSTPONED")
                return assignments # no assignment
            else:
                task, resource = self.take_action(action)

            assignment = ((next((x for x in list(unassigned_tasks) if x.task_type == task), None)), resource)

            available_resources.remove(assignment[1])
            unassigned_tasks.remove(assignment[0])
            busy_resources[assignment[1]] = (assignment[0], self.simulator.now)
            assignments.append(assignment)
        return assignments
        

    def report(self, event):
        pass#print(event)