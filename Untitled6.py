


##imported all the library that we will need

import pandas as pd
import matplotlib.pyplot as plt
import math
import pylab as P
import numpy as np

from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from numpy import array
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score, cross_val_predict



##dataframe is denoted as df and df_1. df_1 is used for visualization purpose

df = pd.read_excel('Dataset_MSCI623.xlsx')
df_1 = pd.read_excel('Dataset_MSCI623.xlsx')
df



## this is to visualize the percentage of missing values in the datasets

import seaborn as sns
fullData_na = (df_1.isnull().sum() / len(df_1)) * 100
fullData_na = fullData_na.drop(fullData_na[fullData_na == 0].index).sort_values(ascending=False)[:30]
missData = pd.DataFrame({'Missing Ratio' :fullData_na})
missData.head(20)
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=fullData_na.index, y=fullData_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
plt.show()



## to handle the missing values

values = {'Gender': 2, 'Habit of having breakfast': 1, 'Place of having breakfast': 4, 'Taste': 2, 'Dirt_Chart': 1, 'Occupation': 2, 'Balanced_Food': 2,'Transportation': 1,'Shopping_days': 3, 'Nutrition_Check': 3,'count_on_having_breakfast_per_week': 7,'Restaurant_food': 2,'Transportation': 1, 'Reason_of_having_breakfast': 3,'Desired_Cost': 1,'Reason_of_not_having_breakfast': 1}
df = df.fillna(value=values)
df


##histogramfor data visualization

df_1.hist(column='Place of having breakfast')
df_1.hist(column='Nutrition_Check' )
df_1.hist(column='Occupation')



print(df.isnull().sum())
df
x = df
y = df.Cereal_Preferred_as_breakfast

print(y)
print(x)


##creating two seperated datasets: with predicted variable and without predicted variable

del x['Cereal_Preferred_as_breakfast']
print(x)



##splitting the data into train and test set

from sklearn.model_selection import train_test_split, cross_val_score
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=90)




##KNN classification

avg_acc = []
for i in range(1,100):
    model = KNeighborsClassifier(n_neighbors=i)
    accuracy = cross_val_score(model, xtrain, ytrain, cv=10)
    avg_acc.append(np.mean(accuracy))
n = np.argmax(avg_acc)
print('Best acc for neighbor = %d' %(n+1))


plt.plot(range(1,100), avg_acc)
plt.title('Average Accuracy for Neighbours')
plt.show()


model = KNeighborsClassifier(n_neighbors=n+1)
model.fit(xtrain, ytrain)
preds = model.predict(xtest)
acc = accuracy_score(ytest, preds)
print('Maximum accuracy= %f' % acc)

from sklearn.metrics import confusion_matrix as cm


preds = model.predict(xtrain)
cm =confusion_matrix(preds, ytrain)
sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');
print(cm)





##Decison tree classification


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=9)
clf.fit(xtrain, ytrain)
preds = clf.predict(xtest)
acc= accuracy_score(ytest, preds)
acc




##Logistic Regression 


from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


logreg = LogisticRegression()
# rfe = RFE(logreg, 10)
# rfe = rfe.fit(xtrain, ytrain)
logreg.fit(xtrain, ytrain)
preds = logreg.predict(xtest)
acc = accuracy_score(ytest, preds)
acc

##Graph for logistic regression

import scipy.stats as stats
#matplotlib.style.use('ggplot')
plt.figure(figsize=(9,9))

def sigmoid(t):                          # Define the sigmoid function
    return (1/(1 + np.e**(-t)))    

plot_range = np.arange(-6, 6, 0.1)       

ytest = sigmoid(plot_range)

# Plot curve
plt.plot(plot_range,   # X-axis range
         ytest, color="red")


from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, preds)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');





##Random Forest


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor

avg_acc = []
for i in range(1,20):
    model = RandomForestClassifier(max_depth=i, random_state=90)
    accuracy = cross_val_score(model, xtrain, ytrain, cv=10)
    avg_acc.append(np.mean(accuracy))
    print('Max Depth: %d, Accuracy: %f\n' %(i,np.mean(accuracy)))

n = np.argmax(avg_acc)
print('Best acc for maxdepth = %d' %n+1)

plt.plot(range(1,20), avg_acc)
plt.show()

model = RandomForestClassifier(max_depth=n, random_state=90)
model.fit(xtrain, ytrain)
preds = model.predict(xtest)
acc = accuracy_score(ytest, preds)
acc




##Support Vector Mechanism


from sklearn.svm import SVR
from sklearn import svm, datasets

C = [0.0001, 0.001, .01, .1, 1, 10, 50, 100, 1000]

avg_acc = []
for i in C:
    model = svm.SVC(kernel='rbf', C=i)
    accuracy = cross_val_score(model, xtrain, ytrain, cv=10)
    avg_acc.append(np.mean(accuracy))
    print('C: %.04f, Accuracy: %f\n' %(i,np.mean(accuracy)))
n = np.argmax(avg_acc)
print('Best acc for C = %f' %C[n])


regressor_svr = svm.SVC(kernel='rbf', C=C[n]).fit(xtrain, ytrain)
preds = regressor_svr.predict(xtest)
acc= accuracy_score(ytest, preds)
acc
plt.show()





##Naive Bayse



from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
preds=model.fit(xtrain, ytrain).predict(xtest)
acc= accuracy_score(ytest, preds)
acc

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, preds)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');





##KMean Clustering


from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets
from sklearn.cluster import KMeans


kmeans_model = KMeans(n_clusters=3, random_state=1).fit(df)
preds=kmeans_model.predict(df)

centers = kmeans_model.cluster_centers_
plt.scatter(centers[:, 1], centers[:, 2], c='black', s=200, alpha=0.5);


