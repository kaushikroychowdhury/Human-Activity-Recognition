import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio
pio.renderers.default='browser'
# import tensorflow as tf
import statsmodels as sm
from sklearn.feature_selection import SelectKBest, f_regression, f_classif


train = pd.read_csv("Human Activity Recognition with Smartphone/train.csv")
test = pd.read_csv("Human Activity Recognition with Smartphone/test.csv")

df1 = train['Activity'].groupby(train['Activity']).count()
labels = train['Activity'].groupby(train['Activity']).count().index.to_list()
values = df1.values.tolist()
fig1 = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
                             insidetextorientation='radial')])
fig1.update_layout(
    title_text="Train Data")

df2 = test['Activity'].groupby(test['Activity']).count()
labels = test['Activity'].groupby(test['Activity']).count().index.to_list()
values = df2.values.tolist()
fig2 = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
                             insidetextorientation='radial')])
fig2.update_layout(
    title_text="Test Data")
# Not much Difference in data Collection

#FacetPlot


laying = train[train['Activity'] == 'LAYING']
sitting = train[train['Activity'] == 'SITTING']
standing = train[train['Activity'] == 'STANDING']
walking = train[train['Activity'] == 'WALKING']
walking_downstairs = train[train['Activity'] == 'WALKING_DOWNSTAIRS']
walking_upstairs = train[train['Activity'] == 'WALING_UPSTAIRS']


facetgrid = sns.FacetGrid(train, hue='Activity', size=6,aspect=2)
facetgrid.map(sns.distplot,'tBodyAccMag-mean()', hist=False).add_legend()
plt.annotate("Stationary Activities", xy=(-0.956,8), xytext=(-0.5, 14), size=20,
            va='center', ha='left',
            arrowprops=dict(arrowstyle="simple",connectionstyle="arc3,rad=0.1"))

plt.annotate("Moving Activities", xy=(0,3), xytext=(0.2, 9), size=20,\
            va='center', ha='left',\
            arrowprops=dict(arrowstyle="simple",connectionstyle="arc3,rad=0.1"))

x1 = walking_upstairs['tBodyAccMag-mean()'].to_numpy()
fig = ff.create_distplot([sitting['tBodyAccMag-mean()']], ['sitting'], show_hist=False)
fig = ff.create_distplot([walking['tBodyAccMag-mean()']], ['walking'], show_hist=False)

# col = train.columns
# x_train = train.iloc[:,:]
# y_train = train.iloc[:,-1]
# x_train.drop(['Activity'], axis=1)

# x = train.drop(['Activity'],axis=1)
# y = train['Activity']

# y_train = pd.Series(y_train).astype('category')
# y_train = y_train.cat.codes


# ........................................... FEATURE SELECTION ........................................................

# selector = SelectKBest(f_classif, k=100)
# selector.fit_transform(x_train, y_train)
# # cols = selector.get_support(indices=True)
# # features_df_new = x_train.iloc[:, cols]
#
# names = x_train.columns.values[selector.get_support()]
# scores = selector.scores_[selector.get_support()]
# names_scores = list(zip(names, scores))
# ns_df = pd.DataFrame(data = names_scores, columns=['Feature_names', 'F_scores'])
# ns_df_sorted = ns_df.sort_values(['F_scores', 'Feature_names'], ascending= False)
# print(ns_df_sorted)
#
# plt.subplots(figsize = (19, 12))
# sns.barplot(x = ns_df_sorted['Feature_names'], y= ns_df_sorted['F_scores'])
# plt.xticks(rotation = 90)
# print(plt.show())

# print(features_df_new.head())

