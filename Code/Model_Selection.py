from HAR_Feature_Selection import *
import umap.umap_ as umap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

### Functions used ..

def ConfusionMatrix_plot(cm, labels, title):
    cm_text = [[str(y) for y in x] for x in cm]
    # set up figure
    fig = ff.create_annotated_heatmap(cm, x=labels, y=labels, annotation_text=cm_text, colorscale='Viridis')

    # add title
    fig.update_layout(title_text=f"<i><b>{title}</b></i>")

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig['data'][0]['showscale'] = True
    fig.show()

def AUROC_curve(model, y, y_scores, title):
    # One hot encode the labels in order to plot them
    y_onehot = pd.get_dummies(y, columns=model.classes_)

    # Create an empty figure, and iteratively add new lines
    # every time we compute a new class
    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    for i in range(y_scores.shape[1]):
        y_true = y_onehot.iloc[:, i]
        y_score = y_scores[:, i]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_score = roc_auc_score(y_true, y_score)

        name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=500
    )
    fig.show()

def f_score(x_train, x_test, y_train, y_test, classifiers):
    for clf in classifiers:

        clf.fit(x_train,y_train)
        y_pred = clf.predict(x_test)
        f = f1_score(y_true=y_test,y_pred=y_pred,average="macro")

        print(f"Score: {round(f,3)*100}% \t Classifier: {clf.__class__.__name__}")

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
labels = ['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']

classifiers = [
    KNeighborsClassifier(5),
    SVC(kernel="rbf"),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GaussianNB(),
    RidgeClassifier(),
    LogisticRegression(max_iter=200)
]
f_score(x_train, x_test, y_train, y_test, classifiers)


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

### Before Feature selection #########

clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)
f1 = f1_score(y_true=y_test,y_pred=y_pred,average="macro")
print("Accuracy for Train data before feature selection")
print(f"Score: {round(f1, 3) * 100}%")
y_scores = clf.predict_proba(x_test)
AUROC_curve(clf, y_test, y_scores, title = 'AUROC Curve for Training Data')

### Accuracy on original Test Data ###
y_pred = clf.predict(xTest)
f2 = f1_score(y_true=yTest,y_pred=y_pred,average="macro")
print("Accuracy for Original Test data before feature selection")
print(f"Score: {round(f2, 3) * 100}%")
y_scores = clf.predict_proba(xTest)
AUROC_curve(clf, yTest, y_scores, title = 'AUROC Curve for Original Test Data')

### After Feature selection #########

features = featureSelection()

x1, xTest1 = train.filter(items=features), test.filter(items=features)
x1 = StandardScaler().fit_transform(x1)
xTest1 = StandardScaler().fit_transform(xTest1)
x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size=0.2, random_state=40)


clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)
f1 = f1_score(y_true=y_test,y_pred=y_pred,average="macro")
cm = confusion_matrix(y_true=y_test,y_pred=y_pred, labels=['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS'])
print("Accuracy for Train data after feature selection")
print(f"Score: {round(f1, 3) * 100}%")
ConfusionMatrix_plot(cm, labels, title = "Confusion Matrix (Training)")    # Confusion matrix for training data
y_scores = clf.predict_proba(x_test)
AUROC_curve(clf, y_test, y_scores, title = 'AUROC Curve for Training Data (After Feature Selection)')


y_pred = clf.predict(xTest1)
f2 = f1_score(y_true=yTest,y_pred=y_pred,average="macro")
cm = confusion_matrix(y_true=yTest,y_pred=y_pred, labels=['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS'])
print("Accuracy for Original Test data after feature selection")
print(f"Score: {round(f2, 3) * 100}%")
ConfusionMatrix_plot(cm, labels, title = "Confusion Matrix (Testing)")    # Confusion matrix for original test data
# AUC on original test data
y_scores = clf.predict_proba(xTest1)
AUROC_curve(clf, yTest, y_scores, title = 'AUROC Curve for Original Test Data (After Feature Selection)')

### Rather than removing Features we can use the reduced features using Advanced Dimensionality Reduction Techniques
trans = umap.UMAP(n_neighbors=5, n_components=5 ,random_state=42).fit(x_train)
clf.fit(trans.embedding_,y_train)
test_embedding = trans.transform(x_test)
y_pred = clf.predict(test_embedding)
f1 = f1_score(y_true=y_test,y_pred=y_pred,average="macro")
print(f"Score: {round(f1, 3) * 100}%")
y_scores = clf.predict_proba(test_embedding)
AUROC_curve(clf, y_test, y_scores, title = 'AUROC Curve for Transformed UMAP Training Data')

# UMAP on original test data

test_embedding = trans.transform(xTest1)
y_pred = clf.predict(test_embedding)
f1 = f1_score(y_true=yTest,y_pred=y_pred,average="macro")
print(f"Score: {round(f1, 3) * 100}%")
y_scores = clf.predict_proba(test_embedding)
AUROC_curve(clf, yTest, y_scores, title = 'AUROC Curve for Transformed UMAP Test Data (Original Test Dataset)')