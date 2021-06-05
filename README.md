# Human-Activity-Recognition

## About the DATASET ...

### Human Activity Recognition Using Smartphones Dataset (Version 1.0)
The experiments have been carried out with a group of 30 volunteers within an age bracket of 19-48 years. Each person performed six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) wearing a smartphone (Samsung Galaxy S II) on the waist. Using its embedded accelerometer and gyroscope, we captured 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz. The experiments have been video-recorded to label the data manually. The obtained dataset has been randomly partitioned into two sets, where 70% of the volunteers was selected for generating the training data and 30% the test data. 

The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration signal, which has gravitational and body motion components, was separated using a Butterworth low-pass filter into body acceleration and gravity. The gravitational force is assumed to have only low frequency components, therefore a filter with 0.3 Hz cutoff frequency was used. From each window, a vector of features was obtained by calculating variables from the time and frequency domain. See 'features_info.txt' for more details. 

#### For each record it is provided:
==================================
- Triaxial acceleration from the accelerometer (total acceleration) and the estimated body acceleration.
- Triaxial Angular velocity from the gyroscope. 
- A 561-feature vector with time and frequency domain variables. 
- Its activity label. 
- An identifier of the subject who carried out the experiment.

#### About Features:
==================
The features selected for this database come from the accelerometer and gyroscope 3-axial raw signals tAcc-XYZ and tGyro-XYZ. These time domain signals (prefix 't' to denote time) were captured at a constant rate of 50 Hz. Then they were filtered using a median filter and a 3rd order low pass Butterworth filter with a corner frequency of 20 Hz to remove noise. Similarly, the acceleration signal was then separated into body and gravity acceleration signals (tBodyAcc-XYZ and tGravityAcc-XYZ) using another low pass Butterworth filter with a corner frequency of 0.3 Hz. 

Subsequently, the body linear acceleration and angular velocity were derived in time to obtain Jerk signals (tBodyAccJerk-XYZ and tBodyGyroJerk-XYZ). Also the magnitude of these three-dimensional signals were calculated using the Euclidean norm (tBodyAccMag, tGravityAccMag, tBodyAccJerkMag, tBodyGyroMag, tBodyGyroJerkMag). 

Finally a Fast Fourier Transform (FFT) was applied to some of these signals producing fBodyAcc-XYZ, fBodyAccJerk-XYZ, fBodyGyro-XYZ, fBodyAccJerkMag, fBodyGyroMag, fBodyGyroJerkMag. (Note the 'f' to indicate frequency domain signals). 

These signals were used to estimate variables of the feature vector for each pattern:  
'-XYZ' is used to denote 3-axial signals in the X, Y and Z directions.

- Features are normalized and bounded within [-1,1].
- Each feature vector is a row on the text file.
- The units used for the accelerations (total and body) are 'g's (gravity of earth -> 9.80665 m/seg2).
- The gyroscope units are rad/seg.

tBodyAcc-XYZ
tGravityAcc-XYZ
tBodyAccJerk-XYZ
tBodyGyro-XYZ
tBodyGyroJerk-XYZ
tBodyAccMag
tGravityAccMag
tBodyAccJerkMag
tBodyGyroMag
tBodyGyroJerkMag
fBodyAcc-XYZ
fBodyAccJerk-XYZ
fBodyGyro-XYZ
fBodyAccMag
fBodyAccJerkMag
fBodyGyroMag
fBodyGyroJerkMag

The set of variables that were estimated from these signals are: 

mean(): Mean value
std(): Standard deviation
mad(): Median absolute deviation 
max(): Largest value in array
min(): Smallest value in array
sma(): Signal magnitude area
energy(): Energy measure. Sum of the squares divided by the number of values. 
iqr(): Interquartile range 
entropy(): Signal entropy
arCoeff(): Autorregresion coefficients with Burg order equal to 4
correlation(): correlation coefficient between two signals
maxInds(): index of the frequency component with largest magnitude
meanFreq(): Weighted average of the frequency components to obtain a mean frequency
skewness(): skewness of the frequency domain signal 
kurtosis(): kurtosis of the frequency domain signal 
bandsEnergy(): Energy of a frequency interval within the 64 bins of the FFT of each window.
angle(): Angle between to vectors.

Additional vectors obtained by averaging the signals in a signal window sample. These are used on the angle() variable:

gravityMean
tBodyAccMean
tBodyAccJerkMean
tBodyGyroMean
tBodyGyroJerkMean

The complete list of variables of each feature vector is available in 'features.txt'

#### Dataset License:
===================
Use of this dataset in publications must be acknowledged by referencing the following publication [1] 

[1] Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013. 

This dataset is distributed AS-IS and no responsibility implied or explicit can be addressed to the authors or their institutions for its use or misuse. Any commercial use is prohibited.

#### Other Related Publications:
==============================
[2] Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra, Jorge L. Reyes-Ortiz.  Energy Efficient Smartphone-Based Activity Recognition using Fixed-Point Arithmetic. Journal of Universal Computer Science. Special Issue in Ambient Assisted Living: Home Care.   Volume 19, Issue 9. May 2013

[3] Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. Human Activity Recognition on Smartphones using a Multiclass Hardware-Friendly Support Vector Machine. 4th International Workshop of Ambient Assited Living, IWAAL 2012, Vitoria-Gasteiz, Spain, December 3-5, 2012. Proceedings. Lecture Notes in Computer Science 2012, pp 216-223. 

[4] Jorge Luis Reyes-Ortiz, Alessandro Ghio, Xavier Parra-Llanas, Davide Anguita, Joan Cabestany, Andreu Català. Human Activity and Motion Disorder Recognition: Towards Smarter Interactive Cognitive Environments. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.  

==================================================================================================
Jorge L. Reyes-Ortiz, Alessandro Ghio, Luca Oneto, Davide Anguita and Xavier Parra. November 2013.


## Exploratory Data Analysis

#### Problem Framework

30 subjects(volunteers) data is randomly split to 70%(21) test and 30%(7) train data.
Each datapoint corresponds one of the 6 Activities.
1. Walking
2. WalkingUpstairs
3. WalkingDownstairs
4. Standing
5. Sitting
6. Lying.

#### Problem Statement
Given a new datapoint we have to predict the Activity

#### Data Cleaning
There is no Duplicate Values in train and test dataset
There is no missing values in this dataset.
Train & Test data is almost balanced.

![Train Dataset Distribution](/Visualization/train_dist.png)
![Test Dataset Distribution](/Visualization/test_dist.png)

##### Data Provided by each subject

![Countplot (Training set)](/Visualization/countplot.png)

#### 1. Featuring Engineering from Domain Knowledge
#### Static and Dynamic Activities

In static activities (sit, stand, lie down) motion information will not be very useful.
In the dynamic activities (Walking, WalkingUpstairs,WalkingDownstairs) motion info will be significant.

#### 2. Stationary and Moving activities are completely different

![Activities](/Visualization/staticvsdynamic.png)
As we can clearly see the difference between stationary activities and Moving Activities as per above pdf distribution.

##### Magnitude of Accelaration can also separate Activities
![Activities](/Visualization/boxtbody.png)
![Activities](/Visualization/boxfbody.png)

###### Observations:

1. If tAccMean is < -0.8 then the Activities are either Standing or Sitting or Laying.
2. If tAccMean is > -0.6 then the Activities are either Walking or WalkingDownstairs or WalkingUpstairs.
3. If tAccMean > 0.0 then the Activity is WalkingDownstairs.
4. We can classify 75% the Acitivity labels with some errors.

##### Position of GravityAccelerationComponants

To understand variation of Gravity Accelaration Components, 3D plot will be easier to interpret as every Smartphone sensors works in 3-Dimensional Plane.

![Sensor Axis](/Visualization/axis_device.png)
source : [Sensor Overview](https://developer.android.com/guide/topics/sensors/sensors_overview#java)

###### Laying
![Laying 3D Plot](/Visualization/laying3d.png)
###### Sitting
![Sitting 3D Plot](/Visualization/sitting3d.png)
###### Standing
![Standing 3D Plot](/Visualization/standing3d.png)
###### Walking
![Walking 3D Plot](/Visualization/walking3d.png)
###### Walking Downstairs
![Walking Downstairs 3D Plot](/Visualization/walkingdownstairs3d.png)
###### Walking Upstairs
![Walking Upstairs 3D Plot](/Visualization/walkingupstairs3d.png)

###### Observations:

1. If angleX,gravityMean > 0 then Activity is Laying.
2. We can classify all datapoints belonging to Laying activity with just a single if else statement.


## Model Selection

In model selection phase, I have train the data into multiple classifier Algorithms. Classifier algorithm are as follows :
1. KNeighborsClassifier
2. SVC
3. Decision-Tree Classifier
4. Random-Forest Classifier
5. GaussianNB
6. Ridge Classifier
7. Logistic Regression

###### Observation

Score: 95.7% 	 Classifier: KNeighborsClassifier\
Score: 98.1% 	 Classifier: SVC\
Score: 94.1% 	 Classifier: DecisionTreeClassifier\
Score: 98.1% 	 Classifier: RandomForestClassifier\
Score: 72.7% 	 Classifier: GaussianNB\
Score: 98.4% 	 Classifier: RidgeClassifier\
Score: 98.9% 	 Classifier: LogisticRegression\


Next I used **Stacking Classifier** because, Stacked generalization consists in stacking the output of individual estimator and use a classifier to compute the final prediction. Stacking allows to use the strength of each individual estimator by using their output as input of a final estimator.

```python
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
```

#### Model Performance: (Observations)

![AUROC Curve Training](/Visualization/trainAUC.png)
![AUROC Curve Testing](/Visualization/testAUC.png)

We got very good Accuracy in multilabel classification by stacking the classifiers.

1. Accuracy for Train data before feature selection\
Score: 98.9%\
2. Accuracy for Original Test data before feature selection\
Score: 95.6%\

## Feature Selection

There are 561 features, I experimented with most important features and also used Advanced Dimensionality Reduction (**UMAP, T-SNE, PCA**)

I used Random-Forest Classifier as the model to select important Features .

```python
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
```

###### Observations

**For 134 features**\
Accuracy for Train data after feature selection\
Score: 99.3%\
Accuracy for Original Test data after feature selection\
Score: 85.1%\

**For 122 features**\
Accuracy for Train data after feature selection\
Score: 99.0%\
Accuracy for Original Test data after feature selection\
Score: 88.0%\

**For 124 features**\
Accuracy for Train data after feature selection\
Score: 99.1%\
Accuracy for Original Test data after feature selection\
Score: 88.7%\

**For 133 features**\
Accuracy for Train data after feature selection\
Score: 98.9%\
Accuracy for Original Test data after feature selection\
Score: 85.1%\


###### Performance metric

![Confusion Matrix Training](/Visualization/cmtraining.png)
![Confusion Matrix Testing](/Visualization/cmtesting.png)
![AUROC Curve Training](/Visualization/trainAUC_AFS.png)
![AUROC Curve Testing](/Visualization/testAUC_AFS.png)

## Dimensionality Reduction for selecting features
The main idea behind redusing dimensions is to reduce all 561 features to n number of components, so that every feature could have their importance in model, rather than only using the selected features.

And also the model will train fast if we reduce the features.
I have used 3 Dimensionality Reduction Algorithm:

1. **PCA** (Principal Component Analysis)
2. **T-SNE** (t-Distributed Stochastic Neighbor Embedding)
3. **UMAP** (Uniform Manifold Approximation and Projection)

#### Visualizing algorithms for this dataset:
## PCA
![PCA](/Visualization/PCA.png)
## T-SNE
![T-SNE](/Visualization/T-SNE.png)
## UMAP
![UMAP](/Visualization/UMAP(Supervised).png)

UMAP performs better than PCA and T-SNE for this dataset, so we can transform our dataset according to the input of UMAP model. And then train the **Stacked Classifier** with the transformed data.

```python
import umap.umap_ as umap
trans = umap.UMAP(n_neighbors=5, n_components=5 ,random_state=42).fit(x_train)
clf.fit(trans.embedding_,y_train)
test_embedding = trans.transform(x_test)
y_pred = clf.predict(test_embedding)
f1 = f1_score(y_true=y_test,y_pred=y_pred,average="macro")
print(f"Score: {round(f1, 3) * 100}%")
```

###### Observation:

umap training   Score: 92.0%\
umap test       Score: 85.9%\

![UMAP train AUROC](/Visualization/trainAUC_UMAP.png)
![UMAP test AUROC](/Visualization/testAUC_UMAP.png)


# Summary

Score: 95.7% 	 Classifier: KNeighborsClassifier\
Score: 98.1% 	 Classifier: SVC\
Score: 94.1% 	 Classifier: DecisionTreeClassifier\
Score: 98.1% 	 Classifier: RandomForestClassifier\
Score: 72.7% 	 Classifier: GaussianNB\
Score: 98.4% 	 Classifier: RidgeClassifier\
Score: 98.9% 	 Classifier: LogisticRegression\


## Stacked Classifier
Accuracy for Train data before feature selection\
Score: 98.9%\
Accuracy for Original Test data before feature selection\
Score: 95.6%\
Accuracy for Train data after feature selection\
Score: 98.9%\
Accuracy for Original Test data after feature selection\
Score: 85.1%\

umap training   Score: 92.0%\
umap test       Score: 85.9%\

__**We get best test accuracy of 95.6% for the Stacked Classifier model using all 561 features.**__
