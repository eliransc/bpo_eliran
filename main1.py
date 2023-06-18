import pandas as pd

from simulator import Simulator
from planners import GreedyPlanner
from planners import planner_Eliran
from planners2 import PPOPlanner
import pickle as pkl
import time
import os
running_time = 24*30
import numpy as np

# Original main ##
def simulate_competition(a1,a2,a3,a4,a5):
    # my_planner = PPOPlanner(model_name = "ppo_masked_long_train_time")
    # my_planner = GreedyPlanner()
    # my_planner = RealGreedyPlanner()
    #my_planner = LPPlanner()
    #my_planner = NoPlanner()

    # 10.879914, 0.475911, 1.456346, 0.928605, 8.479268
    # a1 = A[0]
    # a2 = A[1]
    # a3 = A[2]
    # a4 = A[3]
    # a5 = A[4]

    my_planner = planner_Eliran(a1, a2, a3, a4, a5)

    mod_num = np.random.randint(1, 10000000)
    results = []
    for i in range(15):
        now = time.time()
        simulator = Simulator(planner = my_planner, instance_file="BPI Challenge 2017 - instance.pickle") # running_time = running_time,
        if type(my_planner) == PPOPlanner:
            my_planner.linkSimulator(simulator)
    
        #t1 = time()
        result = simulator.run()[0]
        #print(f'Simulation finished in {time()-t1} seconds')
        print(result)
        path_mod = str(mod_num)+'df_results5.pkl'
        if os.path.exists(path_mod):
            df = pkl.load(open(path_mod, 'rb'))
        else:
            df = pd.DataFrame([])


        curr_ind = df.shape[0]


        results.append(result)
        run_time = time.time() - now
        df.loc[curr_ind, 'a1'] = my_planner.a1
        df.loc[curr_ind, 'a2'] = my_planner.a2
        df.loc[curr_ind, 'a3'] = my_planner.a3
        df.loc[curr_ind, 'a4'] = my_planner.a4
        df.loc[curr_ind, 'a5'] = my_planner.a5

        df.loc[curr_ind, 'instance'] = 2
        df.loc[curr_ind, 'runtime'] = run_time
        df.loc[curr_ind, 'avg_cycle'] = result


        pkl.dump(df, open(path_mod, 'wb'))
        print(df)
    return  -np.array(results).mean()


#    with open("results.txt", "rw") as out_file:
#        for i in results:
#            out_file.write(i)
        

    #print(f'Arrivals: {simulator.case_arrivals}')
    #print(f'Departures: {simulator.case_departures}')
    #print(f'Task arrivals: {simulator.task_arrivals}')
    #print(f'Task departures: {simulator.task_departures}')
    

def main():



    A = [10.879914, 0.475911, 1.456346, 0.928605, 8.479268]

    results = simulate_competition(10.531437856645526, 15.375280086388253, 7.264139715351323, 5.781553323925149, 19.416205458185715)
    pkl.dump(results, open('res.pkl', 'wb'))

if __name__ == "__main__":
    main()