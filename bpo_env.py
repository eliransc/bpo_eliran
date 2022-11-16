import gym
from gym import spaces, Env
import random
import numpy as np
from typing import List, Union

from numpy.core.fromnumeric import shape

from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from simulator import EventType, Event, Task, Problem, MinedProblem, SimulationEvent, TimeUnit

class BPOEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, instance_file="./gym_bpo/envs/BPI Challenge 2017 - instance.pickle"):       
        self.instance_file = instance_file
        
        
        self.reward_interval = 7 #days
        self.count_rewards = 0
        
        
        # replicate the structure of the simulator
        self.events = []

        self.unassigned_tasks = dict()
        self.assigned_tasks = dict()
        self.available_resources = set()
        self.away_resources = []
        self.away_resources_weights = []
        self.busy_resources = dict()
        self.busy_cases = dict()
        self.reserved_resources = dict()
        self.now = 0

        self.finalized_cases = 0
        self.unfinished_cases = 0
        self.total_cycle_time = 0
        self.case_start_times = dict()

        self.problem = MinedProblem.from_file(instance_file)
        self.problem_resource_pool = self.problem.resource_pools

        self.init_simulation()

        #new parameters 
        self.RUNNING_TIME = 24*28
        self.status = "RUNNING"
        self.return_reward = False
        self.total_cycle_time = 0
        self.average_cycle_time = None
        self.previous_average_cycle_time = None #used to split the rewards daily
        self.previous_finalized_cases = 0
        self.current_day = 1
        self.return_flag = False
        
        # parameters needed for masking
        self.task_types = sorted(list(self.problem_resource_pool.keys())) #all task types (should be 7 elements)
        self.resources = sorted(list(set(np.hstack(list(self.problem_resource_pool.values()))))) #all the resources in the problem (should be 145 elements)

        self.inputs = self.resources + self.resources + self.resources + self.task_types #input example 
    
        self.resource_pools_indexes = {}
        self.output = []
        for task, value in self.problem_resource_pool.items():
            indexes = []
            for resource in value:
                indexes.append(self.resources.index(resource)) # Finds and appends the index of a resource
                self.output.append((task, resource))
        
            self.resource_pools_indexes[task] = indexes 
        

        self.output.append((None, None))#(last unitary list is "do nothing")
        # gym specific 
        
        #define lows and highs for different sections of the input
        lows = np.array([0 for x in range(len(self.inputs))])
        
        highs = np.array([1 for x in range(len(self.resources))] + [np.finfo(np.float64).max for x in range(len(self.resources))] + [len(self.task_types) for x in range(len(self.resources))] + [np.finfo(np.float64).max for x in range(len(self.task_types))])

        #first len(resources): ones for available, zero for busy/away; second len(resources): time passed in current task for each resource (zero if free or away); third len(resources): current task a resource is doing (zero if free or away); len(task types): number of each type of task in waiting queue
        self.observation_space = gym.spaces.Box(low=np.float32(lows), high=np.float32(highs))#Box(low=0, high=np.finfo(np.int64).max, shape=shape(self.inputs), dtype=np.int64) #observation space is the concatentenation of resource pools
        self.action_space = spaces.Discrete(len(self.output)) #action space is the cartesian product of tasks and resources in their resource pool

    # NOT USED
    def create_matches_list(self):
        return self.problem_resource_pool
        
    def init_simulation(self):
        # set all resources to available
        for r in self.problem.resources:
            self.available_resources.add(r)

        # generate resource scheduling event to start the schedule
        self.events.append((0, SimulationEvent(EventType.SCHEDULE_RESOURCES, 0, None)))

        # reset the problem
        self.problem.restart()

        # generate arrival event for the first task of the first case
        (t, task) = self.problem.next_case()
        self.events.append((t, SimulationEvent(EventType.CASE_ARRIVAL, t, task)))

    def desired_nr_resources(self):
        return self.problem.schedule[int(self.now % len(self.problem.schedule))]

    def working_nr_resources(self):
        return len(self.available_resources) + len(self.busy_resources) + len(self.reserved_resources)


    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """


        self.take_action(action)

        #print(f"ACTION TAKEN: {action}")

        reward = self.get_reward()

        self.return_flag = False

        while not self.return_flag and self.status!="FINISHED": #action_masks always contains at least a 1 (the do nothing action)
            self.simulator_step()

        if self.status=="FINISHED":
            #reward -= self.unfinished_cases #unifinished cases at the end are bad!
            print(f"LAST REWARD: {reward}")

        return self.getState(), reward, self.status == "FINISHED", {}

    def getState(self):

        #av_resources_ones = list(np.where(np.isin(np.asarray(self.resources), self.available_resources), 1, 0))
        av_resources_ones = [1 if x in self.available_resources else 0 for x in self.resources]
         
        #get current time in task for each resource (x[1] is the processing start time)
        busy_resources_times = [self.now - self.busy_resources[x][1] if x in self.busy_resources else 0 for x in self.resources]
        
        #get current task type for each busy resource
        busy_resources_tasks = [self.task_types.index(self.busy_resources[x][0].task_type) + 1 if x in self.busy_resources else 0 for x in self.resources]


        #task_types_num = [np.count_nonzero(el in [o.task_type for o in self.unassigned_tasks.values()]) for el in self.task_types]
        task_types_num =  [np.sum(el in [o.task_type for o in self.unassigned_tasks.values()]) for el in self.task_types]

        return np.asarray(av_resources_ones + busy_resources_times + busy_resources_tasks + task_types_num)
    

    def reset(self):
        self.__init__()
        print("Environment reset \n")
        return self.getState()

    def render(self, mode='human', close=False):
        print(f"Average reward: {self.average_cycle_time}")

    def take_action(self, action):
        assignments = [self.output[action]]
        self.check_outputs(assignments)


    #def _choose_next_state(self) -> None:
    #   self.state = self.getState()
    

    def get_reward(self):
         
        self.count_rewards += 1
        if (self.status == "RUNNING" and self.return_reward):
            
            self.return_reward = False
            self.current_reward = self.finalized_cases if self.previous_finalized_cases == 0 else self.finalized_cases - self.previous_finalized_cases
            
            self.previous_finalized_cases = self.finalized_cases
            #print(f"Current reward: {self.current_reward}")
            #print(f"Current reward number: {self.count_rewards}")

            return self.current_reward
        elif (self.status == "FINISHED"): #we need to give reward also at the last ts!
            self.return_reward = False
            self.current_reward = self.finalized_cases - self.previous_finalized_cases - self.unfinished_cases #unfinished cases should not contribute to the reward
            
            self.previous_finalized_cases = self.finalized_cases
        else:
            return 0

    def get_reward_old(self):
         
        self.count_rewards += 1
        if (self.status == "RUNNING" and self.return_reward) or (self.status == "FINISHED"): #we need to give reward also at the last ts!
            
            self.return_reward = False


            if self.previous_average_cycle_time != None:
                self.average_cycle_time = (self.total_cycle_time - self.previous_total_cycle_time)/(self.finalized_cases - self.previous_finalized_cases)
            else:
                self.average_cycle_time = self.total_cycle_time/self.finalized_cases

            #current_reward = 1/self.average_cycle_time
            current_reward = -self.average_cycle_time

            self.previous_average_cycle_time = self.average_cycle_time                
            self.previous_total_cycle_time = self.total_cycle_time
            self.previous_finalized_cases = self.finalized_cases

            #print(f"Current reward: {current_reward}")
            #print(f"Current reward number: {self.count_rewards}")
            return current_reward
        
        else:
            return 0

    #return value is False if the simulation must continue, True if we reached a terminal state
    def simulator_step(self):
        return_flag = False
        while self.now <= self.RUNNING_TIME:
            # we return only when a planning event happens
            return_flag = False

            # get the first event e from the events
            event = self.events.pop(0)
            # t = time of e
            self.now = event[0]
            event = event[1]
        
            # if e is an arrival event:
            if event.event_type == EventType.CASE_ARRIVAL:
                self.case_start_times[event.task.case_id] = self.now
                #self.planner.report(Event(event.task.case_id, None, self.now, None, EventType.CASE_ARRIVAL))
                # add new task
                #self.planner.report(Event(event.task.case_id, event.task, self.now, None, EventType.TASK_ACTIVATE))
                self.unassigned_tasks[event.task.id] = event.task
                self.busy_cases[event.task.case_id] = [event.task.id]
                # generate a new planning event to start planning now for the new task
                self.events.append((self.now, SimulationEvent(EventType.PLAN_TASKS, self.now, None, nr_tasks=len(self.unassigned_tasks), nr_resources=len(self.available_resources))))
                # generate a new arrival event for the first task of the next case
                (t, task) = self.problem.next_case()
                self.events.append((t, SimulationEvent(EventType.CASE_ARRIVAL, t, task)))
                self.events.sort()

            # if e is a start event:
            elif event.event_type == EventType.START_TASK:
                #self.planner.report(Event(event.task.case_id, event.task, self.now, event.resource, EventType.START_TASK))
                # create a complete event for task
                t = self.now + self.problem.processing_time(event.task, event.resource)
                self.events.append((t, SimulationEvent(EventType.COMPLETE_TASK, t, event.task, event.resource)))
                self.events.sort()
                # set resource to busy
                del self.reserved_resources[event.resource]
                self.busy_resources[event.resource] = (event.task, self.now)

            # if e is a complete event:
            elif event.event_type == EventType.COMPLETE_TASK:
                #self.planner.report(Event(event.task.case_id, event.task, self.now, event.resource, EventType.COMPLETE_TASK))
                # set resource to available, if it is still desired, otherwise set it to away
                del self.busy_resources[event.resource]
                if self.working_nr_resources() <= self.desired_nr_resources():
                    self.available_resources.add(event.resource)
                else:
                    self.away_resources.append(event.resource)
                    self.away_resources_weights.append(self.problem.resource_weights[self.problem.resources.index(event.resource)])
                # remove task from assigned tasks
                del self.assigned_tasks[event.task.id]
                self.busy_cases[event.task.case_id].remove(event.task.id)
                # generate unassigned tasks for each next task
                for next_task in self.problem.next_tasks(event.task):
                    #self.planner.report(Event(event.task.case_id, next_task, self.now, None, EventType.TASK_ACTIVATE))
                    self.unassigned_tasks[next_task.id] = next_task
                    self.busy_cases[event.task.case_id].append(next_task.id)
                if len(self.busy_cases[event.task.case_id]) == 0:
                    #self.planner.report(Event(event.task.case_id, None, self.now, None, EventType.COMPLETE_CASE))
                    self.events.append((self.now, SimulationEvent(EventType.COMPLETE_CASE, self.now, event.task)))
                # generate a new planning event to start planning now for the newly available resource and next tasks
                self.events.append((self.now, SimulationEvent(EventType.PLAN_TASKS, self.now, None, nr_tasks=len(self.unassigned_tasks), nr_resources=len(self.available_resources))))
                self.events.sort()

            # if e is a schedule resources event: move resources between available/away,
            # depending to how many resources should be available according to the schedule.
            elif event.event_type == EventType.SCHEDULE_RESOURCES:
                assert self.working_nr_resources() + len(self.away_resources) == len(self.problem.resources)  # the number of resources must be constant
                assert len(self.problem.resources) == len(self.problem.resource_weights)  # each resource must have a resource weight
                assert len(self.away_resources) == len(self.away_resources_weights)  # each away resource must have a resource weight
                if len(self.away_resources) > 0:  # for each away resource, the resource weight must be taken from the problem resource weights
                    i = random.randrange(len(self.away_resources))
                    assert self.away_resources_weights[i] == self.problem.resource_weights[self.problem.resources.index(self.away_resources[i])]
                required_resources = self.desired_nr_resources() - self.working_nr_resources()
                if required_resources > 0:
                    # if there are not enough resources working
                    # randomly select away resources to work, as many as required
                    for i in range(required_resources):
                        random_resource = random.choices(self.away_resources, self.away_resources_weights)[0]
                        # remove them from away and add them to available resources
                        away_resource_i = self.away_resources.index(random_resource)
                        del self.away_resources[away_resource_i]
                        del self.away_resources_weights[away_resource_i]
                        self.available_resources.add(random_resource)
                    # generate a new planning event to put them to work
                    self.events.append((self.now, SimulationEvent(EventType.PLAN_TASKS, self.now, None, nr_tasks=len(self.unassigned_tasks), nr_resources=len(self.available_resources))))
                    self.events.sort()
                elif required_resources < 0:
                    # if there are too many resources working
                    # remove as many as possible, i.e. min(available_resources, -required_resources)
                    nr_resources_to_remove = min(len(self.available_resources), -required_resources)
                    resources_to_remove = random.sample(self.available_resources, nr_resources_to_remove)
                    for r in resources_to_remove:
                        # remove them from the available resources
                        self.available_resources.remove(r)
                        # add them to the away resources
                        self.away_resources.append(r)
                        self.away_resources_weights.append(self.problem.resource_weights[self.problem.resources.index(r)])
                # plan the next resource schedule event
                self.events.append((self.now + 1, SimulationEvent(EventType.SCHEDULE_RESOURCES, self.now + 1, None)))

            # if e is a planning event: do assignment
            elif event.event_type == EventType.PLAN_TASKS:
                # there only is an assignment if there are free resources and tasks
                if sum(self.define_action_masks())>1:
                    #assignments = self.planner.plan(self.available_resources.copy(), list(self.unassigned_tasks.values()), self.problem_resource_pool)
                    self.return_flag = True
                
                    #previously some checks on assignments were here, now moved outside
                
                

            # if e is a complete case event: add to the number of completed cases
            elif event.event_type == EventType.COMPLETE_CASE:
                self.total_cycle_time += self.now - self.case_start_times[event.task.case_id]
                self.finalized_cases += 1
                self.return_reward = True
                #self.return_flag = True #we return just to give a reward

            #if self.now > self.current_day*24*self.reward_interval: #give reward every self.reward_interval days
            #    self.return_reward = True
            #    self.current_day += 1

            if self.return_flag:
                return None # the episode is not over yet
        
        self.status = "FINISHED" # the episode is over

        self.unfinished_cases = 0
        for busy_tasks in self.busy_cases.values():
            if len(busy_tasks) > 0:
                if busy_tasks[0] in self.unassigned_tasks:
                    busy_case_id = self.unassigned_tasks[busy_tasks[0]].case_id
                else:
                    busy_case_id = self.assigned_tasks[busy_tasks[0]][0].case_id
                if busy_case_id in self.case_start_times:
                    start_time = self.case_start_times[busy_case_id]
                    if start_time <= self.RUNNING_TIME:
                        self.total_cycle_time += self.RUNNING_TIME - start_time
                        self.finalized_cases += 1
                        self.unfinished_cases += 1
        
        self.average_cycle_time = self.total_cycle_time/self.finalized_cases                
        print(f"AVERAGE CYCLE TIME: {self.average_cycle_time}")
        return None
     


    # check outputs
    def check_outputs(self, assignments_str):
        #retrieve the first task in available_tasks with given name
        assignments = [((next((x for x in list(self.unassigned_tasks.values()) if x.task_type == task), None)), resource) for (task, resource) in assignments_str]
        assignments_size = len(assignments) # we compute the size to assess if an assignment was skipped

        # for each newly assigned task:
        moment = self.now
        for (task, resource) in assignments:
            if task==None and resource==None:
                if (assignments_size) > 1: #print only when a postponement happened when other actions were available
                    print("WARNING: postponed action.")
                continue
            if task not in self.unassigned_tasks.values():
                print("ERROR: trying to assign a task that is not in the unassigned_tasks.")
                return None, "ERROR: trying to assign a task that is not in the unassigned_tasks."
            if resource not in self.available_resources:
                print("ERROR: trying to assign a resource that is not in available_resources.")
                return None, "ERROR: trying to assign a resource that is not in available_resources."
            if resource not in self.problem_resource_pool[task.task_type]:
                print("ERROR: trying to assign a resource to a task that is not in its resource pool.")
                return None, "ERROR: trying to assign a resource to a task that is not in its resource pool."
            # create start event for task
            self.events.append((moment, SimulationEvent(EventType.START_TASK, moment, task, resource)))
            # assign task
            del self.unassigned_tasks[task.id]
            self.assigned_tasks[task.id] = (task, resource, moment)
            # reserve resource
            self.available_resources.remove(resource)
            self.reserved_resources[resource] = (task, moment)
        #self.events.sort()

    # define mask based on current environment state (only the 3 vectors that are also known at inference time!)
    def define_action_masks(self) -> List[bool]:
        state = self.getState()
        mask = [0 for _ in range(len(self.output))]
        for i in range(len(self.resources)*3, len(self.resources)*3 + len(self.task_types)):
            if state[i] > 0: # Check for available tasks
                task = self.inputs[i] # Get task string
                for resource_index in self.resource_pools_indexes[task]:
                    if state[resource_index] > 0:
                        mask[self.output.index((task, self.resources[resource_index]))] = 1
        
        mask[-1] = 1 #"do nothing" is always allowed                
        self.invalid_actions = list(map(bool,mask))
        return self.invalid_actions

    
    def action_masks(self) -> List[bool]:
        simple_mask = self.define_action_masks()
        return simple_mask

    ## NOT USED
    #check if there are assignments possible
    def check_assignments(self):
        for i in self.unassigned_tasks.values().unique():
            if self.available_resources.any() in self.resource_pools[i]:
                return True
        return False