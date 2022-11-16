from simulator import Simulator
from planners import GreedyPlanner
from planners import planner_Eliran
from planners2 import PPOPlanner
import os
import pickle5 as pkl
import numpy as np
import pandas as pd

running_time = 24 * 100


def single_train(running_time, my_planner, bpi_file):
    simulator = Simulator(running_time=running_time, planner=my_planner, instance_file=bpi_file)
    result = simulator.run()

    return result


# Original main
def simulate_competition():
    # my_planner = PPOPlanner(model_name = "ppo_masked_long_train_time")
    # my_planner = GreedyPlanner()
    # my_planner = RealGreedyPlanner()
    # my_planner = LPPlanner()
    # my_planner = NoPlanner()
    my_planner = planner_Eliran()
    df_main_path = 'df_results.pkl'



    print('first settings')
    result1 = [single_train(running_time, my_planner, "BPI Challenge 2017 - instance.pickle") for ind in range(1)]
    print('second settings')
    result2 = [single_train(running_time, my_planner, "BPI Challenge 2017 - instance 2.pickle") for ind in range(1)]

    print(result1, result2)
    print(np.array(result1 + result2).mean())

    if os.path.exists(df_main_path):
        df = pkl.load(open(df_main_path, 'rb'))
    else:
        df = pd.DataFrame([])

    curr_ind = df.shape[0]

    df.loc[curr_ind, 'a1'] = my_planner.a1
    df.loc[curr_ind, 'a2'] = my_planner.a2
    df.loc[curr_ind, 'a3'] = my_planner.a3
    df.loc[curr_ind, 'a4'] = my_planner.a4
    df.loc[curr_ind, 'a5'] = my_planner.a5
    df.loc[curr_ind, 'set1'] = np.array(result1).mean()
    df.loc[curr_ind, 'set2'] = np.array(result2).mean()
    df.loc[curr_ind, 'total'] = np.array(np.array(result1 + result2)).mean()

    pkl.dump(df, open(df_main_path, 'wb'))
    print(df)




#    with open("results.txt", "rw") as out_file:
#        for i in results:
#            out_file.write(i)


# print(f'Arrivals: {simulator.case_arrivals}')
# print(f'Departures: {simulator.case_departures}')
# print(f'Task arrivals: {simulator.task_arrivals}')
# print(f'Task departures: {simulator.task_departures}')


def main():
    simulate_competition()


if __name__ == "__main__":
    main()