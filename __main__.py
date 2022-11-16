from simulator import Simulator
from planners import GreedyPlanner
from planners import planner_Eliran
from planners2 import PPOPlanner
import pickle5 as pkl
import time

running_time = 24*365


# Original main ##
def simulate_competition():
    # my_planner = PPOPlanner(model_name = "ppo_masked_long_train_time")
    #my_planner = GreedyPlanner()
    #my_planner = RealGreedyPlanner()
    #my_planner = LPPlanner()
    #my_planner = NoPlanner()
    my_planner = planner_Eliran()

    now = time.time()
    results = []
    for i in range(1):
        simulator = Simulator(running_time = running_time, planner = my_planner, instance_file="BPI Challenge 2017 - instance.pickle")

        if type(my_planner) == PPOPlanner:
            my_planner.linkSimulator(simulator)
    
        #t1 = time()
        result = simulator.run()
        #print(f'Simulation finished in {time()-t1} seconds')
        print(result)
        results.append(result)
        run_time = time.time() - now
        pkl.dump((run_time, result), open('result1.pkl', 'wb'))
        print(run_time)

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