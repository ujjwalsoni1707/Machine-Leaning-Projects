import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Machine Learning Algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('heart_data.csv')
#print(df.columns)
#print(df.describe)

#bar plot
rcParams['figure.figsize'] = 8,6
plt.bar([1,0], df['target'].value_counts(), color = ['red', 'green'])
plt.xticks([0, 1])
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Count of each Target Class')


#Data Preprocessing
df = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
df[columns_to_scale] = standardScaler.fit_transform(df[columns_to_scale])

#Splitting the data into Train and Test
y = df['target']
X = df.drop(['target'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

#Applying KNeghborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 8)
knn_classifier.fit(X_train, y_train)
knn_score=knn_classifier.score(X_test, y_test)

#Applying Support Vector Machine
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
svc_classifier = SVC(kernel = kernels[0])
svc_classifier.fit(X_train, y_train)
svc_score=svc_classifier.score(X_test, y_test)

#Applying DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(max_features = 18, random_state = 0)
dt_classifier.fit(X_train, y_train)
dt_score=dt_classifier.score(X_test, y_test)

#Applying RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 100, random_state = 0)
rf_classifier.fit(X_train, y_train)
rf_score=rf_classifier.score(X_test, y_test)

print("KNeighborsClassifier",knn_score)
print("Support Vector Classifier",svc_score)
print("DecisionTreeClassifier",dt_score)
print("RandomForestClassifier",rf_score)