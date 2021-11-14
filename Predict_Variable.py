from sklearn.preprocessing import PolynomialFeatures as PF
from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DEGREE1 = 4 
DEGREE2 = 4 
DEGREE3 = 2

data = pd.read_csv('data1.csv')
lm = linear_model.LinearRegression()


X_people = data.iloc[:,0:1]
X_people = PF(degree=DEGREE1).fit_transform(X_people)
y_people = data.iloc[:,6:7]
people_model = lm.fit(X_people,y_people)

lm = linear_model.LinearRegression()
X_house = data.iloc[:,0:1]
X_house = PF(degree=DEGREE2).fit_transform(X_house)
y_house = data.iloc[:,1:6]
house_model = lm.fit(X_house,y_house)

pred_x_area = list(range(50,280))
pred_y_people = people_model.predict(PF(degree=DEGREE1).fit_transform(pd.DataFrame(pred_x_area)))
pred_y_people = [round(i[0]) for i in pred_y_people]
pred_y_house = house_model.predict(PF(degree=DEGREE2).fit_transform(pd.DataFrame(pred_x_area)))

data = pd.read_csv('data2.csv')
lm=linear_model.LinearRegression()
Xx = data.iloc[:,0:9]
Xx = PF(degree=DEGREE3).fit_transform(Xx)
y = data.iloc[:,9:26]
model = lm.fit(Xx,y)

pred_x = []
for i in range(50,280):
    i -= 50
    pred_x.append([pred_x_area[i],pred_y_house[i][0],pred_y_house[i][1],pred_y_house[i][2],pred_y_house[i][3],pred_y_house[i][4],pred_y_people[i]*0.2,pred_y_people[i]*0.4,pred_y_people[i]*0.4])

pred_y = model.predict(PF(degree=DEGREE3).fit_transform(pd.DataFrame(pred_x)))

# ASK AREA
# area = eval(input("AREA: "))

# for i in pred_y[area-50]:
#     print(round(i),',',end='')

y = [i[16] for i in np.array(y).tolist()]
pred_y = [i[16] for i in pred_y]
X_people = data.iloc[:,0:1]
plt.plot(pred_x_area,pred_y,color='orange')
plt.scatter(np.array(X_people).tolist(),y,color='orange')
plt.ylabel('Vacuum Cleaner')
plt.xlabel('Area of house')
plt.show()


print(pred_y)