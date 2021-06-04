### Testing which advanced dimensionality reduction method is better of this data ###
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
sns.set()
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'

train = pd.read_csv("Human Activity Recognition with Smartphone/train.csv")
train = shuffle(train)
LE = LabelEncoder()
train['code'] = LE.fit_transform(train['Activity'])
# train = train.iloc[:200]
x = train.drop(['subject', 'Activity'],axis=1)
y = train['code']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20 ,random_state=42)

test = pd.read_csv("Human Activity Recognition with Smartphone/test.csv")
test = shuffle(test)
LE = LabelEncoder()
test['code'] = LE.fit_transform(test['Activity'])
x = test.drop(['subject', 'Activity'],axis=1)
y = test['code']

def plot_2d(component1, component2):
    fig = go.Figure(data=go.Scatter(
        x=component1,
        y=component2,
        mode='markers',
        marker=dict(
            size=20,
            color=y,  # set color equal to a variable
            colorscale='Rainbow',  # one of plotly colorscales
            showscale=True,
            line_width=1
        )
    ))
    fig.update_layout(margin=dict(l=100, r=100, b=100, t=100), width=2000, height=1200)
    fig.layout.template = 'plotly_dark'

    fig.show()


def plot_3d(component1, component2, component3):
    fig = go.Figure(data=[go.Scatter3d(
        x=component1,
        y=component2,
        z=component3,
        mode='markers',
        marker=dict(
            size=10,
            color=y,  # set color to an array/list of desired values
            colorscale='Rainbow',  # choose a colorscale
            opacity=1,
            line_width=1
        )
    )])


    # tight layout
    fig.update_layout(margin=dict(l=50, r=50, b=50, t=50), width=1800, height=1000)
    fig.layout.template = 'plotly_dark'

    fig.show()

x = StandardScaler().fit_transform(x)

pca = PCA().fit(x)
exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
fig1 = px.area(
    x=range(1, exp_var_cumul.shape[0] + 1),
    y=exp_var_cumul,
    labels={"x": "# Components", "y": "Explained Variance"}
)
# import matplotlib.pyplot as plt
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')           ## Matplotlib

###################################################### PCA ##################   Visualization of data #################
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principal = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3'])

plot_2d(principalComponents[:, 0],principalComponents[:, 1])
plot_3d(principalComponents[:, 0],principalComponents[:, 1],principalComponents[:, 2])


# ################################################### T-SNE ############################################################
pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(x)
tsne = TSNE(random_state = 42, n_components=3,verbose=0, perplexity=40, n_iter=400).fit_transform(pca_result_50)

plot_2d(tsne[:, 0],tsne[:, 1])
plot_3d(tsne[:, 0],tsne[:, 1],tsne[:, 2])

# ################################################# UMAP #################################################################

reducer = umap.UMAP(random_state=42,n_components=3)
embedding = reducer.fit_transform(x)

plot_2d(reducer.embedding_[:, 0],reducer.embedding_[:, 1])
plot_3d(reducer.embedding_[:, 0],reducer.embedding_[:, 1],reducer.embedding_[:, 2])

# ################################################ LDA ###################################################################

X_LDA = LDA(n_components=3).fit_transform(x,y)
# plot_2d(X_LDA[:, 0],X_LDA[:, 1])                          ## Not Working because dimension of LDA is only 1
# plot_3d(X_LDA[:, 0],X_LDA[:, 1],X_LDA[:, 2])              ## instead of 3 components, it returned only 1


# As we can see UMAP outperforms other techniques for this Dataset ..
# https://umap-learn.readthedocs.io/en/latest/supervised.html#using-labels-to-separate-classes-supervised-umap
s = time.time()
embedding = umap.UMAP(n_neighbors=20).fit_transform(x, y=y)
e = time.time()
print(f"time consumed: {round(e-s,3)}")
plot_2d(embedding[:, 0],embedding[:, 1])




