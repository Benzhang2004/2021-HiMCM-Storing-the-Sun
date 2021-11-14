from sklearn.preprocessing import PolynomialFeatures as PF
from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DEGREE = 4

data = pd.read_csv('data1.csv')
lm = linear_model.LinearRegression()


X_house = data.iloc[:,0:1]
X_house = PF(degree=DEGREE).fit_transform(X_house)
y_house = data.iloc[:,1:6]
people_model = lm.fit(X_house,y_house)

pred_x = list(range(50,280))
pred_y = people_model.predict(PF(degree=DEGREE).fit_transform(pd.DataFrame(pred_x)))

print(pred_y)
y =  np.array(y_house).tolist()
pred_y = [i[4] for i in pred_y]
print(pred_y)
# xpred = PF(degree=DEGREE).fit_transform(pd.DataFrame([148]))

X_house = data.iloc[:,0:1]
plt.plot(pred_x,pred_y)
plt.scatter(np.array(X_house).tolist(),[i[4] for i in np.array(y_house).tolist()])
plt.ylabel('Toilet')
plt.xlabel('Area of house')
plt.show()