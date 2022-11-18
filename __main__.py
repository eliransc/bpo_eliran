import pandas as pd

from simulator import Simulator
from planners import GreedyPlanner
from planners import planner_Eliran
from planners2 import PPOPlanner
import pickle5 as pkl
import time
import os
running_time = 24*365


# Original main ##
def simulate_competition():
    # my_planner = PPOPlanner(model_name = "ppo_masked_long_train_time")
    # my_planner = GreedyPlanner()
    # my_planner = RealGreedyPlanner()
    #my_planner = LPPlanner()
    #my_planner = NoPlanner()
    my_planner = planner_Eliran()



    results = []
    for i in range(3):
        now = time.time()
        simulator = Simulator( planner = my_planner, instance_file="BPI Challenge 2017 - instance 2.pickle")

        if type(my_planner) == PPOPlanner:
            my_planner.linkSimulator(simulator)
    
        #t1 = time()
        result = simulator.run()
        #print(f'Simulation finished in {time()-t1} seconds')
        print(result)
        if os.path.exists('df_results1.pkl'):
            df = pkl.load(open('df_results1.pkl', 'rb'))
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


        pkl.dump(df, open('df_results1.pkl', 'wb'))
        print(df)

#    with open("results.txt", "rw") as out_file:
#        for i in results:
#            out_file.write(i)
        

    #print(f'Arrivals: {simulator.case_arrivals}')
    #print(f'Departures: {simulator.case_departures}')
    #print(f'Task arrivals: {simulator.task_arrivals}')
    #print(f'Task departures: {simulator.task_departures}')
    

def main():
    simulate_competition()

if __name__ == "__main__":
    main()