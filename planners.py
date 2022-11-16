from abc import ABC, abstractmethod
from sb3_contrib.ppo_mask import MaskablePPO, MlpPolicy
#from bpo_env import BPOEnv
import numpy as np
import json
import pandas as pd
from itertools import permutations, combinations
import time
from simulator import Simulator

class planner_Eliran:

    def __init__(self):

        self.case_num = np.random.randint(1,10000000)
        self.a1 = 10.879914
        self.a2 = 0.475911
        self.a3 = 1.456346
        self.a4 = 0.928605
        self.a5 = 8.479268
        self.df_mean_var = pd.DataFrame(columns=['resource'])

        all_cols = ['task', 'W_Complete application', 'W_Call after offers',
                    'W_Validate application', 'W_Call incomplete files',
                    'W_Handle leads', 'W_Assess potential fraud',
                    'W_Shortened completion', 'complete_case']
        num_cols = len(all_cols)
        initial_df_vals = np.zeros((num_cols - 2, num_cols))
        self.df_freq_transition = pd.DataFrame(initial_df_vals, columns=all_cols)

        for task_ind, task_ in enumerate(all_cols):
            if (task_ind > 0) and (task_ind < num_cols - 1):
                self.df_freq_transition.loc[task_ind - 1, 'task'] = task_

        self.df = pd.DataFrame([])


    def check_if_there_is_possible_match(self, available_resources, unassigned_tasks, resource_pool):

        for task in unassigned_tasks:
            for resource in available_resources:
                if resource in resource_pool[task.task_type]:
                    return True
        return False



    def get_task_out_prob(self, df_trans, task_):
        curr_ind = df_trans.loc[df_trans['task'] == task_, :].index[0]
        if df_trans.iloc[curr_ind, 1:].sum()>0:
            return df_trans.iloc[curr_ind, -1] / df_trans.iloc[curr_ind, 1:].sum()
        else:
            return 0

    def give_task_ranking(self, df_mean_var, avail_res, task):

        ######################################################
        ##### Begining: Ranking tasks within resource ########
        ######################################################
        if (df_mean_var.shape[0] > 0) & ('mean_'+str(task) in list(df_mean_var.columns)):
            df_res_ranking = pd.DataFrame([])
            for ind_res, res in enumerate(avail_res):
                curr_ind = df_res_ranking.shape[0]
                if df_mean_var.loc[df_mean_var['resource'] == res, 'mean_' + str(task)].shape[0] > 0:
                    if df_mean_var.loc[df_mean_var['resource'] == res, 'mean_' + str(task)].item()>0:
                        df_res_ranking.loc[curr_ind, 'resource'] = res
                        df_res_ranking.loc[curr_ind, 'task_mean'] = df_mean_var.loc[
                            df_mean_var['resource'] == res, 'mean_' + str(task)].item()

            if df_res_ranking.shape[0] > 0:
                df_res_ranking = df_res_ranking.sort_values(by='task_mean').reset_index()
                df_res_ranking['Ranking'] = np.arange(df_res_ranking.shape[0])
                return df_res_ranking


    def give_resource_ranking(self, df_mean_var, resource, unassigned_tasks_):

        ######################################################
        ##### Begining: Ranking tasks within resource ########
        ######################################################

        if (df_mean_var.shape[0] > 0) & (resource in list( df_mean_var['resource'])):  # if df_mean_var initiated and if we have knowegle about resource
            # task_names = [col for col in df_mean_var.columns if col.startswith('mean')]
            df_ranking_tasks = pd.DataFrame([])
            unassigned = [task for task in unassigned_tasks_ if 'mean_' + task in df_mean_var.columns]
            if len(unassigned) > 0:
                for ind, curr_task in enumerate(unassigned):
                    df_ranking_tasks.loc[ind, 'task_name'] = curr_task
                    df_ranking_tasks.loc[ind, 'task_mean'] = df_mean_var.loc[
                        df_mean_var['resource'] == resource, 'mean_'+curr_task].item()

                df_ranking_task = df_ranking_tasks.loc[df_ranking_tasks['task_mean'] > 0, :].sort_values(
                    by='task_mean').reset_index()
                df_ranking_task['Ranking'] = np.arange(df_ranking_task.shape[0])


                return df_ranking_task

        #################################################
        ##### End: Ranking tasks within resource ########
        #################################################

    def plan(self, available_resources, unassigned_tasks, resource_pool):

        a1 = self.a1
        a2 = self.a2
        a3 = self.a3
        a4 = self.a4
        a5 = self.a5


        if self.df.shape[0]>0:
            curr_df_status = self.df.index[-1]
        else:
            curr_df_status = 0



        assignments = []
        # assign the first unassigned task to the first available resource, the second task to the second resource, etc.


        unassigned_tasks_ = [task.task_type for task in unassigned_tasks]

        dict_ranking_tasks = {}
        for resource in available_resources:
            dict_ranking_tasks[resource] = self.give_resource_ranking(self.df_mean_var, resource, set(unassigned_tasks_))

        dict_ranking_resource = {}

        for task in set(unassigned_tasks_):
            dict_ranking_resource[task] = self.give_task_ranking(self.df_mean_var, available_resources, task)


        df_combs_score = pd.DataFrame([])

        for task in set(unassigned_tasks_):
            for resource in available_resources:
                if resource in resource_pool[task]:

                    mean_val = -1
                    var_val = -1
                    if self.df_mean_var.shape[0] > 0:
                        if 'mean_' + task in self.df_mean_var.columns:
                            if self.df_mean_var.loc[self.df_mean_var['resource'] == resource, 'mean_' + task].shape[
                                0] > 0:
                                if self.df_mean_var.loc[
                                    self.df_mean_var['resource'] == resource, 'mean_' + task].item() > 0:
                                    mean_val = self.df_mean_var.loc[
                                        self.df_mean_var['resource'] == resource, 'mean_' + task].item()
                                    var_val = self.df_mean_var.loc[
                                        self.df_mean_var['resource'] == resource, 'var_' + task].item()

                    curr_df = dict_ranking_resource[task]
                    res_rank = -1
                    if not curr_df is None:
                        if not (curr_df == None).all()[0]:
                            if curr_df.loc[curr_df['resource'] == resource, 'Ranking'].shape[0] > 0:
                                res_rank = curr_df.loc[curr_df['resource'] == resource, 'Ranking'].item()

                    curr_df = dict_ranking_tasks[resource]
                    task_rank = -1
                    if not curr_df is None:
                        if not (curr_df == None).all()[0]:
                            if curr_df.loc[curr_df['task_name'] == task, 'Ranking'].shape[0]:
                                task_rank = curr_df.loc[curr_df['task_name'] == task, 'Ranking'].item()



                    if self.df_freq_transition.shape[0]>0:

                        prob = self.get_task_out_prob(self.df_freq_transition, task)
                        if not prob >= 0:
                            prob = -1
                    else:
                        prob = -1


                    curr_ind = df_combs_score.shape[0]
                    df_combs_score.loc[curr_ind, 'resource'] = resource
                    df_combs_score.loc[curr_ind, 'task'] = task
                    df_combs_score.loc[curr_ind, 'mean_val'] = mean_val
                    df_combs_score.loc[curr_ind, 'var_val'] = var_val
                    df_combs_score.loc[curr_ind, 'res_rank'] = res_rank
                    df_combs_score.loc[curr_ind, 'task_rank'] = task_rank
                    df_combs_score.loc[curr_ind, 'prob'] = prob
                    df_combs_score.loc[curr_ind, 'tot_score'] = a1*mean_val*+a2*var_val+a3*res_rank+a4*task_rank-a5*prob




        unassigned_tasks_ = [task.task_type for task in unassigned_tasks]

        dict_ranking_tasks = {}
        for resource in available_resources:
            dict_ranking_tasks[resource] = self.give_resource_ranking(self.df_mean_var, resource, set(unassigned_tasks_))

        dict_ranking_resource = {}

        for task in set(unassigned_tasks_):
            dict_ranking_resource[task] = self.give_task_ranking(self.df_mean_var, available_resources, task)


        df_sched_score = pd.DataFrame([])

        for task in set(unassigned_tasks_):
            for resource in available_resources:
                if resource in resource_pool[task]:

                    mean_val = -1
                    var_val = -1
                    if self.df_mean_var.shape[0] > 0:
                        if 'mean_' + task in self.df_mean_var.columns:
                            if self.df_mean_var.loc[self.df_mean_var['resource'] == resource, 'mean_' + task].shape[0] > 0:
                                if self.df_mean_var.loc[
                                    self.df_mean_var['resource'] == resource, 'mean_' + task].item() > 0:
                                    mean_val = self.df_mean_var.loc[
                                        self.df_mean_var['resource'] == resource, 'mean_' + task].item()
                                    var_val = self.df_mean_var.loc[
                                        self.df_mean_var['resource'] == resource, 'var_' + task].item()

                    curr_df = dict_ranking_resource[task]
                    res_rank = -1
                    if not curr_df is None:
                        if not (curr_df == None).all()[0]:
                            if curr_df.loc[curr_df['resource'] == resource, 'Ranking'].shape[0] > 0:
                                res_rank = curr_df.loc[curr_df['resource'] == resource, 'Ranking'].item()

                    curr_df = dict_ranking_tasks[resource]
                    task_rank = -1
                    if not curr_df is None:
                        if not (curr_df == None).all()[0]:
                            if curr_df.loc[curr_df['task_name'] == task, 'Ranking'].shape[0]:
                                task_rank = curr_df.loc[curr_df['task_name'] == task, 'Ranking'].item()



                    if self.df_freq_transition.shape[0]:
                        prob = self.get_task_out_prob(self.df_freq_transition, task)
                        if not prob >= 0:
                            prob = -1
                    else:
                        prob = -1


                    curr_ind = df_sched_score.shape[0]
                    df_sched_score.loc[curr_ind, 'resource'] = resource
                    df_sched_score.loc[curr_ind, 'task'] = task
                    df_sched_score.loc[curr_ind, 'mean_val'] = mean_val
                    df_sched_score.loc[curr_ind, 'var_val'] = var_val
                    df_sched_score.loc[curr_ind, 'res_rank'] = res_rank
                    df_sched_score.loc[curr_ind, 'task_rank'] = task_rank
                    df_sched_score.loc[curr_ind, 'prob'] = prob
                    df_sched_score.loc[curr_ind, 'tot_score'] = a1*mean_val+a2*var_val+a3*res_rank+a4*task_rank+a5*prob

        if (df_sched_score.shape[0] > 0 and curr_df_status > 2000):  # if there is at least one task resource combination in  df_sched_score
            df_sched_score = df_sched_score.sort_values(by='tot_score').reset_index()


            while self.check_if_there_is_possible_match(available_resources, unassigned_tasks, resource_pool):
                for ind in range(df_sched_score.shape[0]):

                    task = df_sched_score.loc[ind, 'task']
                    res = df_sched_score.loc[ind, 'resource']
                    inds_tasks = [task_ind for task_ind in range(len(unassigned_tasks)) if (unassigned_tasks[task_ind].task_type==task)]
                    if len(inds_tasks) > 0:
                        if res in available_resources:
                            if res in resource_pool[task]:
                                assignments.append((unassigned_tasks[inds_tasks[0]], res))

                                available_resources.remove(res)
                                unassigned_tasks.pop(inds_tasks[0])

                                break

        else:

            for task in unassigned_tasks:
                for resource in available_resources:
                    if resource in resource_pool[task.task_type]:

                        available_resources.remove(resource)
                        assignments.append((task, resource))
                        break

        return assignments


    def report(self, event):


        curr_ind = self.df.shape[0]
        if curr_ind > 1000:
            self.df = self.df.iloc[1:,:]
            curr_ind = self.df.index[-1]+1

        self.df.loc[curr_ind,'case_id'] = event.case_id
        self.df.loc[curr_ind, 'task'] = str(event.task)
        self.df.loc[curr_ind, 'timestamp'] = event.timestamp
        self.df.loc[curr_ind, 'date_time'] = str(event).split('\t')[2]
        self.df.loc[curr_ind, 'resource'] = event.resource
        self.df.loc[curr_ind, 'lifecycle_state'] = str(event.lifecycle_state)


        # if a new task is activated or a case is completed
        if str(event.lifecycle_state) == 'EventType.TASK_ACTIVATE' or str(event.lifecycle_state) == 'EventType.COMPLETE_CASE':



            if str(event.lifecycle_state) == 'EventType.TASK_ACTIVATE':
                prev_lifecycle = self.df.loc[self.df.index[-2],'lifecycle_state']
                if prev_lifecycle == 'EventType.COMPLETE_TASK':
                    prev_task = self.df.loc[self.df.index[-2], 'task']
                    curr_task = self.df.loc[self.df.index[-1], 'task']
                    self.df_freq_transition.loc[self.df_freq_transition['task']==prev_task,curr_task] = self.df_freq_transition.loc[self.df_freq_transition['task']==prev_task,curr_task].item() + 1


            elif str(event.lifecycle_state) == 'EventType.COMPLETE_CASE':
                prev_task = self.df.loc[self.df.index[-2], 'task']
                self.df_freq_transition.loc[self.df_freq_transition['task']==prev_task,'complete_case'] = self.df_freq_transition.loc[self.df_freq_transition['task']==prev_task,'complete_case'].item() + 1



        if str(event.lifecycle_state) == 'EventType.COMPLETE_TASK': # if a task just completed
            resource = event.resource  # Give the current resource type
            task = str(event.task)  # Give the current task type
            # The index of the task start time event
            start_ind = self.df.loc[(self.df['task'] == task) & (self.df['lifecycle_state'] == 'EventType.START_TASK'), :].index[-1]
            # Computing the service time
            ser_time = event.timestamp- self.df.loc[start_ind,'timestamp']
            curr_resources = self.df_mean_var['resource'].unique() # All possible resources that were
            if resource in curr_resources:  # if the current resource already been added in the past
                get_ind = self.df_mean_var.loc[self.df_mean_var['resource']==resource,:].index[0] # if so, find its row
            else:
                get_ind = self.df_mean_var.shape[0]
                self.df_mean_var.loc[get_ind, 'resource'] = resource

            if 'count_' + task in  self.df_mean_var.columns:  # if this column exists
                get_count_value = self.df_mean_var.loc[get_ind, 'count_' + task]  # get the count so far
            else:
                get_count_value = 0
            if get_count_value > 0:  # if the count already took place
                get_mean_value = self.df_mean_var.loc[get_ind, 'mean_' + task]  # get the mean service time so far
                get_var_value = self.df_mean_var.loc[get_ind, 'var_' + task]  # get the service variance so far
                curr_mean = (get_count_value*get_mean_value+ser_time)/(get_count_value+1) # compute the new service mean
                if get_count_value > 1:  # For computing variance we need two values at least
                    squer_sum = get_var_value*(get_count_value-1)+get_count_value*get_mean_value**2 # updating variance
                    get_count_value += 1  # updating count
                    curr_var = (squer_sum+ser_time**2-get_count_value*curr_mean**2)/(get_count_value-1) # updating variance

                else:  # if this is the second time we get this combination (task, resource)
                    # the variace is updated accordingly
                    get_count_value = 2 # the count must be 2
                    curr_var = (get_mean_value-curr_mean)**2+(ser_time-curr_mean)**2 # by defintion
            else:
                get_count_value = 1
                curr_mean = ser_time
                curr_var = 0  # it is not really zero but not defined yet.


            # Updating df_mean_var table
            self.df_mean_var.loc[get_ind, 'mean_' + task] = curr_mean
            self.df_mean_var.loc[get_ind, 'var_' + task] = curr_var
            self.df_mean_var.loc[get_ind, 'count_' + task] = get_count_value


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


# Greedy assignment
class GreedyPlanner(Planner):
    """A :class:`.Planner` that assigns tasks to resources in an anything-goes manner."""

    def plan(self, available_resources, unassigned_tasks, resource_pool, last_assignment):
        assignments = []
        available_resources = available_resources.copy()
        i = 0
        # assign the first unassigned task to the first available resource, the second task to the second resource, etc.
        for task in unassigned_tasks:
            for resource in available_resources:
                if resource in resource_pool[task.task_type]:
                    available_resources.remove(resource)
                    assignments.append((task, resource))
                    break
        return assignments

    def report(self, event):
        pass#print(event)


# Greedy assignment
class RealGreedyPlanner(Planner):
    """A :class:`.Planner` that assigns tasks to resources in a "kind of greedy" way."""

    def __init__(self) -> None:
        with open('./gym_bpo/envs/distributions_standardized.json', 'r') as fp:
            self.distributions = json.load(fp)


    def plan(self, available_resources, unassigned_tasks, resource_pool):
        assignments = []
        available_resources = available_resources.copy()
        # assign the first unassigned task to the first available resource, the second task to the second resource, etc.
        for task in unassigned_tasks:
            for resource in self.distributions[task.task_type].keys():
                if resource in available_resources and resource in resource_pool[task.task_type]:
                    available_resources.remove(resource)
                    assignments.append((task, resource))
                    break
        return assignments

    def report(self, event):
        pass #print(event)

from collections import Counter
class LPPlanner(Planner):
    """A :class:`.Planner` that assigns tasks to resources due to a (possibly truncated) llp."""

    def __init__(self) -> None:
        with open('./gym_bpo/envs/distributions_standardized.json', 'r') as fp:
            self.distributions = json.load(fp)

    def plan(self, available_resources, unassigned_tasks, resource_pool):

        print(f'Number of resources: {len(available_resources)}. Number of tasks: {len(unassigned_tasks)}.') 
        available_resources = available_resources.copy()

        rows = []
        for resource in available_resources:
            row = [resource]
            for task in unassigned_tasks:
                if resource in self.distributions[task.task_type].keys():
                    row.append(self.distributions[task.task_type][resource])
                else:
                    row.append(0)
            rows.append(row)

        df = pd.DataFrame(rows, columns=['resource'] + [task.id for task in unassigned_tasks])
        task_dict = {task.id:task for task in unassigned_tasks}
        df = df.set_index('resource')
 
        assignments = []
        df = df.drop(df[df.sum(axis=1) == 0].index)
        zero_colums = (df.sum(axis=0) == 0)        
        df = df.drop(columns=zero_colums[zero_colums].index)
        df.replace(to_replace = 0, value = float('inf'), inplace=True)
        print(Counter([task.task_type for task in unassigned_tasks]))
        #print(list(df.index), list(df.columns))
        if len(df.columns) > 10:
            df = df.drop(columns=df.sample(n=len(df.columns)-10,axis='columns').columns)

        if len(df.index) != 0 or len(df.columns) != 0:
            if len(df.index) > len(df.columns):
                perms_rows = list(combinations(df.index, len(df.columns)))
                perms_columns = list(permutations(df.columns, len(df.columns)))
            else: # len(df.index) < len(df.columns)
                perms_rows = list(permutations(df.index, len(df.index)))
                perms_columns = list(combinations(df.columns, len(df.index)))
        else: # no values, so the list will be empty
            return []


        result = [(x, y) for x in perms_rows for y in perms_columns]

        prev_sum = float("inf")

        for el in result:
            df_new = df.reindex(list(el[0]))
            df_new = df_new[list(el[1])]

            # if len(df_new) > len(df_new.columns):
            #     n = len(df_new.columns)
            #     df_temp = df_new.iloc[:n, :]
            # else:
            #     n = len(df_new)
            #     df_temp = df_new.iloc[:, :n]
            df_temp = df_new.copy()
            sum_diag = sum([df_temp.iloc[i, i] for i in range(len(df_temp))])
            
            if sum_diag < prev_sum:
                #takes the row name and column name of elements on the diagonal
                assignments = []
                for i in range(len(df_temp)):
                    print((task_dict[df_temp.columns[i]], df_temp.index[i]))
                    assignments.append((task_dict[df_temp.columns[i]], df_temp.index[i]))
                    prev_sum = sum_diag

        for assignment in assignments:
            available_resources.remove(assignment[1])
        print(f'Number of assignments: {len(assignments)}')
        return assignments

    def report(self, event):
        pass #print(event)

class NoPlanner():
    def plan(self, available_resources, unassigned_tasks, resource_pool):
        return []
        
    def report(self, event):
        pass #print(event)

# DRL based assignment
class PPOPlanner(Planner):
    """A :class:`.Planner` that assigns tasks to resources following policy dictated by (pretrained) DRL algorithm."""

    def __init__(self) -> None:
        self.model = MaskablePPO.load("ppo_masked_test")
        self.resources = None
        self.task_types = None
        self.inputs = None
        self.output = []
        self.resource_pools_indexes = {}
        self.reward_interval = 3

        self.simulator = None

    #pass the simulator for bidirectional communication
    def linkSimulator(self, simulator):
        self.simulator = simulator
  
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
            self.output.append((None, None)) #do nothing action
        
        assignments = []

        
        available_resources = available_resources.copy() #is this useful?
        unassigned_tasks = unassigned_tasks.copy() # I dont know but let's do it
        busy_resources = self.simulator.busy_resources.copy()

        # PROBLEM: like this, if a resource and a task are available but the net tells to skip the assignment
        while (sum(self.getActionMasks(self.getState(available_resources, unassigned_tasks, busy_resources))) > 1): #the do nothing action is always available
            obs = self.getState(available_resources, unassigned_tasks, busy_resources)
            action_masks = self.getActionMasks(obs)
            action, _states = self.model.predict(obs, action_masks=action_masks)
            task, resource = self.take_action(action)

            if task != None and resource != None:

                #print(f"AVAILABLE RESOURCES: {available_resources}")
                #print(f"UNASSIGNED TASKS: {unassigned_tasks}")
                #print(f"NUMBER OF POSSIBLE ASSIGNMENTS: {sum(self.getActionMasks(self.getState(available_resources, unassigned_tasks, busy_resources))) - 1}")
                assignment = ((next((x for x in list(unassigned_tasks) if x.task_type == task), None)), resource)

                available_resources.remove(assignment[1])
                unassigned_tasks.remove(assignment[0])
            
                assignments.append(assignment)

                busy_resources[assignment[1]] = (assignment[0], self.simulator.now)

            #self.busy_resources.append(assignment[1])
            else:
                #print(f"AVAILABLE RESOURCES: {available_resources}")
                #print(f"UNASSIGNED TASKS: {unassigned_tasks}")
                #print(f"NUMBER OF POSSIBLE ASSIGNMENTS: {sum(self.getActionMasks(self.getState(available_resources, unassigned_tasks, busy_resources))) - 1}")
                #print("DECISION POSTPONED")
                break #if a do nothing happens, all subsequent assignments are also dropped

        #print(f"ASSIGNMENTS: {assignments}")
        return assignments
        

    def report(self, event):
        pass#print(event)


def main():


    my_planner = planner_Eliran()
    simulator = Simulator(my_planner)
    result = simulator.run()
    print(result)

if __name__ == "__main__":


    main()