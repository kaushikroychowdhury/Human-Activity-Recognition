from HAR_Feature_Selection import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.utils import shuffle
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("Human Activity Recognition with Smartphone/train.csv")
test = pd.read_csv("Human Activity Recognition with Smartphone/test.csv")
train = shuffle(train)
test = shuffle(test)
x = train.drop(['Activity'],axis=1)
x = StandardScaler().fit_transform(x)
y = train['Activity']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)

xTest = test.drop(['Activity'],axis=1)
xTest = StandardScaler().fit_transform(xTest)
yTest = test['Activity']

classifiers = [
    KNeighborsClassifier(5),
    SVC(kernel="rbf"),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GaussianNB(),
    RidgeClassifier(),
    LogisticRegression(max_iter=200)
]

def f_score(x_train, x_test, y_train, y_test):
    for clf in classifiers:

        clf.fit(x_train,y_train)
        y_pred = clf.predict(x_test)
        f = f1_score(y_true=y_test,y_pred=y_pred,average="macro")

        print(f"Score: {round(f,3)*100}% \t Classifier: {clf.__class__.__name__}")

f_score(x_train, x_test, y_train, y_test)


estimators = [
        ('RFC' ,RandomForestClassifier(n_estimators=500, random_state = 42)),
        ('KNC', KNeighborsClassifier(5)),
        ('DTC', DecisionTreeClassifier()),
        ('SVC', SVC(kernel="rbf")),
        ('RC',  RidgeClassifier()),
]

clf = StackingClassifier(
    estimators=estimators,
    final_estimator=GradientBoostingClassifier()
)

###  Accuracy for Train data
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
f1 = f1_score(y_true=y_test,y_pred=y_pred,average="macro")
print("Accuracy for Train data before feature selection")
print(f"Score: {round(f1, 3) * 100}%")

### Accuracy on original Test Data ###
y_pred = clf.predict(xTest)
f2 = f1_score(y_true=yTest,y_pred=y_pred,average="macro")
print("Accuracy for Original Test data before feature selection")
print(f"Score: {round(f2, 3) * 100}%")

### After Feature selection #########

features = featureSelection()

x1, xTest1 = train.filter(items=features), test.filter(items=features)
x1 = StandardScaler().fit_transform(x1)
xTest1 = StandardScaler().fit_transform(xTest1)
x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size=0.2, random_state=40)

clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
f1 = f1_score(y_true=y_test,y_pred=y_pred,average="macro")
print("Accuracy for Train data after feature selection")
print(f"Score: {round(f1, 3) * 100}%")

y_pred = clf.predict(xTest1)
f2 = f1_score(y_true=yTest,y_pred=y_pred,average="macro")
print("Accuracy for Original Test data after feature selection")
print(f"Score: {round(f2, 3) * 100}%")
