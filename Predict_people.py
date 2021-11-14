from sklearn.preprocessing import PolynomialFeatures as PF
from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DEGREE = 4

data = pd.read_csv('data1.csv')
lm = linear_model.LinearRegression()


X_people = data.iloc[:,0:1]
X_people = PF(degree=DEGREE).fit_transform(X_people)
y_people = data.iloc[:,6:7]
people_model = lm.fit(X_people,y_people)

pred_x = list(range(50,280))
pred_y = people_model.predict(PF(degree=DEGREE).fit_transform(pd.DataFrame(pred_x)))


y =  np.array(y_people).tolist()
pred_y = [i[0] for i in pred_y]
print(pred_y)
# xpred = PF(degree=DEGREE).fit_transform(pd.DataFrame([148]))

X_people = data.iloc[:,0:1]
plt.plot(pred_x,pred_y)
plt.scatter(np.array(X_people).tolist(),np.array(y_people).tolist())
plt.xlabel('Area of house')
plt.ylabel('people live in house')
plt.show()