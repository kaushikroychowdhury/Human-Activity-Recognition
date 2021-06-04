# ........................................... FEATURE SELECTION Using Random Forest Classifier........................................................

import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


def featureSelection():
    train = pd.read_csv("Human Activity Recognition with Smartphone/train.csv")
    x = train.drop(['subject', 'Activity'], axis=1)
    y = train['Activity']

    selectFeature = SelectFromModel(RandomForestClassifier())
    selectFeature.fit(x, y)

    features1 = x.columns[(selectFeature.get_support())]
    return features1

# print(len(features1))
# print(features1)

# 135 features selected

