import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score                  #for accuracy
import matplotlib.pyplot as plt                             #for histogram

df=pd.read_csv("Social_Network_Ads.csv")
x1=df["Age"]
x2=df["EstimatedSalary"]

label=df["Purchased"]

features=list(zip(x1,x2))

x_train, x_test, y_train, y_test=train_test_split(features, label, test_size=0.20)

mymodel=KNeighborsClassifier(n_neighbors = 3)
mymodel.fit(x_train, y_train)

y_predict=mymodel.predict(x_test)

accuracy=accuracy_score(y_test, y_predict)
print("")
print("ACCURACY = ")
print(accuracy)

#Histogram
plt.hist(df['Purchased'], bins=10)
plt.grid(True)
plt.show()

#ScatterPlot
plt.figure(figsize=(8, 6))
plt.scatter(df['Age'], df['EstimatedSalary'], color='purple', alpha=0.5)
plt.title('Scatter Plot of Age vs Estimated Salary')
plt.xlabel('Age')
plt.grid(True)
plt.show()
