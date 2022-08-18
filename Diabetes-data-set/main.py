import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()

diabetes_X= diabetes.data
# [:,np.newaxis,2]
print(diabetes_X)

# spliting
diabetes_X_train=diabetes_X[:-30]
diabetes_X_test=diabetes_X[-30:]

diabetes_y_train=diabetes.target[:-30]
diabetes_y_test=diabetes.target[-30:]

model = linear_model.LinearRegression()

model.fit(diabetes_X_train,diabetes_y_train)
diabetes_y_predicted=model.predict(diabetes_X_test)

print("Mean squared error is: ", mean_squared_error(diabetes_y_test,diabetes_y_predicted))
# y=w1+w2x
print("Weights: ",model.coef_)
print("Intercepts: ",model.intercept_)

# plt.scatter(diabetes_X_test,diabetes_y_test)
# plt.plot(diabetes_X_test,diabetes_y_predicted)
# plt.show()






# x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,42,52,61,37,82,91]
# y=x[:-5]
# z=x[-5:]
# print(y)
# print(z)