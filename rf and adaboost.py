# import package and data
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,roc_curve,roc_auc_score,classification_report
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier

data = pd.read_excel('test_3980_train.xlsx')
data.dropna(inplace=True)
Y=data['loan_status_']
X=data.loc[:,'loan_amnt':'pub_rec_bankruptcies']
Y_train=pd.read_csv('Y_train.csv')
X_train=pd.read_csv('X_train.csv')
Y_test = pd.read_csv('Y_test.csv')
X_test=pd.read_csv('X_test.csv')

# adaboost
clf_tree = DecisionTreeClassifier(max_depth = 1, random_state = 1)
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME.R",
                         n_estimators=20, learning_rate=0.8)
bdt.fit(X_train,Y_train)
print('ada:' )
print(bdt.score(X_test,Y_test))
Y_pred = bdt.predict(X_test)
print(classification_report(Y_pred, Y_test))
fpr, tpr, threshold=roc_curve(Y_test,Y_pred)

# plot confusion matrix and results
labels = [ ' Charged Off','Fully Paid']
cm = confusion_matrix( Y_pred,Y_test)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix ')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.show()
print(confusion_matrix(Y_pred,Y_test))
print(classification_report(Y_test, Y_pred))

#random forest
rf0 = RandomForestClassifier(oob_score=True, random_state=100,max_features=0.20)
rf0.fit(X_train,Y_train)

#plot a single tree
estimator = rf0.estimators_[0]
from sklearn.tree import export_graphviz
dotdata=export_graphviz(estimator, out_file='tree.dot',
                feature_names = X_test.columns.values.tolist(),
                rounded = True, proportion = False,
                precision = 2, filled = True)

#plot importrances
colNames=list(X_train.columns)
impt=rf0.feature_importances_
c = list(zip(colNames,impt))
c.sort(reverse = True,key=lambda x:x[1])
colNames[:],impt[:] = zip(*c)
print(rf0.feature_importances_)
plt.bar(colNames,impt)
import pylab as pl
pl.xticks(rotation=90)

#plot confusion matrix and results
print('rf: ')
print(rf0.score(X_test,Y_test))
Y_pred = rf0.predict(X_test)
fpr, tpr, threshold=roc_curve(Y_test,Y_pred)
print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_pred, Y_test))

labels = [ ' Charged Off','Fully Paid']
cm = confusion_matrix(Y_pred,Y_test)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix ')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.show()



