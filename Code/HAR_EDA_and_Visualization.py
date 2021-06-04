import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
pio.renderers.default='browser'


train = pd.read_csv("Human Activity Recognition with Smartphone/train.csv")
test = pd.read_csv("Human Activity Recognition with Smartphone/test.csv")

print(train.isna().sum())    ## Checking for null values

train.describe()            ## Descriptive stats
test.describe()

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
# Data is fairly balanced

fig3 = px.histogram(train, x='subject', color="Activity", barmode='group')
fig3.update_layout(title_text="Countplot",
    xaxis = dict(
        tickmode = 'linear'
    )
)                                 # Countplot shows collection of data from each subject for each activity

###################### Relation between Statics and Dynamic activities ##############################################
laying = train[train['Activity'] == 'LAYING'].iloc[:986]
sitting = train[train['Activity'] == 'SITTING'].iloc[:986]
standing = train[train['Activity'] == 'STANDING'].iloc[:986]
walking = train[train['Activity'] == 'WALKING'].iloc[:986]
walking_downstairs = train[train['Activity'] == 'WALKING_DOWNSTAIRS'].iloc[:986]
walking_upstairs = train[train['Activity'] == 'WALKING_UPSTAIRS'].iloc[:986]

fig5 = px.box(train, x='Activity', y='tBodyAccMag-mean()', color='Activity')
fig5.add_shape(type="line",
    x0=-1, y0=-0.4, x1=6, y1=-0.4,
    line=dict(
        color="Grey",
        width=1,
        dash="dashdot",
    ))                           # Below the value -0.4 for 'tBodyAccMag-mean()', all activities are stationary,
                                 # above that moving activities

fig6 = px.box(train, x='Activity', y='fBodyAccMag-mean()', color='Activity')


hist_data = [laying['tBodyAccMag-mean()'], sitting['tBodyAccMag-mean()'], standing['tBodyAccMag-mean()'], walking['tBodyAccMag-mean()'],
             walking_downstairs['tBodyAccMag-mean()'], walking_upstairs['tBodyAccMag-mean()']]
group_labels = ['Laying', 'Sitting', 'Standing', 'Walking', 'Walking Downstairs', 'Waking Upstairs']
fig4 = ff.create_distplot(hist_data, group_labels, show_hist=False)
fig4.add_annotation(x=-0.9, y=13,
            text="Stationary Activity",
            showarrow=False,
            arrowhead=1, align='right')
fig4.add_annotation(x=-0.1, y=4,
            text="Moving Activity",
            showarrow=True,
            arrowhead=1)
############################################ 3d plots #############################################################

laying = train[train['Activity'] == 'LAYING'].iloc[:200]
sitting = train[train['Activity'] == 'SITTING'].iloc[:200]
standing = train[train['Activity'] == 'STANDING'].iloc[:200]
walking = train[train['Activity'] == 'WALKING'].iloc[:200]
walking_downstairs = train[train['Activity'] == 'WALKING_DOWNSTAIRS'].iloc[:200]
walking_upstairs = train[train['Activity'] == 'WALKING_UPSTAIRS'].iloc[:200]


def plot3d (activity):
    x = activity['angle(X,gravityMean)']
    y = activity['angle(Y,gravityMean)']
    z = [activity['angle(Z,gravityMean)']] * 200
    fig7 = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig7.update_layout(title=activity['Activity'].values[0], autosize=True,
                       margin=dict(l=65, r=50, b=65, t=90))
    fig7.show()

plot3d(laying)
plot3d(sitting)
plot3d(standing)
plot3d(walking)
plot3d(walking_upstairs)
plot3d(walking_downstairs)
fig1.show()
fig2.show()
fig3.show()
fig4.show()
fig5.show()
fig6.show()
############## all PDF's for showing moving and stationary activities (seaborn) ######

# facetgrid = sns.FacetGrid(train, hue='Activity', size=6,aspect=2)
# facetgrid.map(sns.distplot,'tBodyAccMag-mean()', hist=False).add_legend()
# plt.annotate("Stationary Activities", xy=(-0.956,8), xytext=(-0.5, 14), size=20,
#             va='center', ha='left',
#             arrowprops=dict(arrowstyle="simple",connectionstyle="arc3,rad=0.1"))
#
# plt.annotate("Moving Activities", xy=(0,3), xytext=(0.2, 9), size=20,
#             va='center', ha='left',
#             arrowprops=dict(arrowstyle="simple",connectionstyle="arc3,rad=0.1"))
