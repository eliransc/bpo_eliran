from simulator import Simulator
from planners import PPOPlanner



my_planner = PPOPlanner()
simulator = Simulator(my_planner)
result = simulator.run()
print(result)
