import pandas as pd
from sko.GA import GA
import numpy as np

import matplotlib.pyplot as plt

CONSUME_ELLAPSE = 2

battery_data = pd.read_csv('data4.csv')
item_power_data = pd.read_csv('data5.csv')
power_up_data = pd.read_csv('data6.csv')

power_up_data = np.array(power_up_data.iloc[:,1:]).tolist()
power_up_data = [[1 if j>20 else 0 for j in i] for i in power_up_data]
item_power_data = np.array(item_power_data.iloc[:,1:]).tolist()
item_power_data = [i[0] for i in item_power_data]

needs = eval(input("Input Item Needs: ")) # array(17)
peak_power = np.matmul(np.multiply(np.array(needs),item_power_data),np.array(power_up_data)).tolist()

energy_consume = sum(peak_power)*CONSUME_ELLAPSE
each_reach_peak = np.array(battery_data.iloc[:,2]).tolist()
each_capacity = np.array(battery_data.iloc[:,3]).tolist()
each_cost = np.array(battery_data.iloc[:,1]).tolist()
# print(each_capacity)
AREA = 111
MAX_COST = 1000*(sum(each_cost))
print(max(peak_power))
print(energy_consume)
def func(x1,x2,x3,x4,x5):
    if each_reach_peak[0]*x1+each_reach_peak[1]*x2+each_reach_peak[2]*x3+each_reach_peak[3]*x4+each_reach_peak[4]*x5<max(peak_power):
        return MAX_COST
    elif each_capacity[0]*x1+each_capacity[1]*x2+each_capacity[2]*x3+each_capacity[3]*x4+each_capacity[4]*x5<energy_consume-AREA*7:
        return MAX_COST
    elif (x3+x4)/(x1+x2+x3+x4+x5)<0.02:
        return MAX_COST
    else:
        return each_cost[0]*x1+each_cost[1]*x2+each_cost[2]*x3+each_cost[3]*x4+each_cost[4]*x5

ga = GA(func=func,n_dim=5, lb=[0,0,0,0,0], ub=[1000,1000,1000,1000,1000], max_iter=500,precision=[1,1,1,1,1])
best_x,best_y = ga.run()
print(best_x,best_y)



# result=MAX_COST
# result_x=[]
# for a in range(0,101):
#     for b in range(0,101):
#         for c in range(0,101):
#             for d in range(0,101):
#                 for e in range(0,101):
#                     num = func(a,b,c,d,e)
#                     if num<result:
#                         result=num
#                         result_x=[a,b,c,d,e]

# print("Result: ",result)
# print("Result_x: ",result_x)

