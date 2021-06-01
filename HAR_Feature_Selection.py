# ........................................... FEATURE SELECTION ........................................................

import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
pio.renderers.default='browser'

train = pd.read_csv("Human Activity Recognition with Smartphone/train.csv")
x = train.drop(['subject', 'Activity'],axis=1)
y = train['Activity']

selectFeature = SelectFromModel(RandomForestClassifier())
selectFeature.fit(x,y)

features1 = x.columns[(selectFeature.get_support())]
print(len(features1))
print(features1)

# 135 features selected

