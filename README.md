# ML-Project
Activity Recognition from wearable Mounted Accelerometer Data Set

I took the 15 csv files and convert into dataframes using pandas and i have concatinated frames by adding column named person with numbers.
I took sample from total set of dataset
i have checked null values and checked data types.
1.Populating a dataframe with means of each of the label. I call it means.  Taking each of the labels and looking at the columns:
In this the y_acceleration is always more irrespective of the activity done. Looks like activity number 2 (Standing up, Walking and Going up\down the stairs is the least for all parameters). The label looks very inconclusive considering it has all the activities in it.
The highest value for z_acceleration is when the person is going up/down the stairs (label 5)
The hypothesis from y_acceleration mean is very inconclusive
x_acceleration is the highest when the person in walking and going up/down the stairs (label 5,6)
2. I removed outliers and checked if there any change in mean. There was no point removing the outliers as the average of each of the acceleration's are still the same. Thus we are going to proceed with the df_sa where the outliers were not removed
IN PIE CHART:
Here we can see that most of the values given are for
label 1 - Working on computer
label 7 - Talking while standing
label 4 - Walking
IN KDE PLOT of x_acceleration:
we can see that x_acceleration is the between 1600 - 2300. This is true for all the labels. However as we can see the pink line represents label 7 which is talking while standing. This has the highest probability for higher values.
IN KDE PLOT of y_acceleration:
 We can see that most of the y_acceleration values lie from 2300 - 2700. Label 1 and 7 have the highest spike in the graph
IN KDE PLOT of z_acceleration:
we can see, the majority of the values are between 1800 to 2200. The highest spike of accelerometer is noticed when standing up.
Exploration between attributes:
1) We can notice from the scatter plots of each of the acceleration and between each other, there is not much correlation neither positively nor negatively
 2) Comparing the labels and the acceleration : we can notice sitting in the computer (label 1) has the most acceleration followed by talking while standing (label 7)
IN HEATMAP:
We can see from the heat map that there is close to 0.5 correlation on an average between the accelerations and close to 0.1 correlation between the labels and the acceleration.
Data Modeling
The sampled dataset which is present right now is a numerical descriptive feature. So thereby scaling descriptive features, we can then train them on a classification model.
Fitting a classifier: Decision Tree Classifier and KNN Classifier
*Hyper-Parameter Tuning:
K-Nearest-Neighbor Classifier: Feeding in 1-7 nearest neighbors and giving Euclidean, Manhattan and Minkowski distance as p values.
We can see that the mean CV score always increases with respect to number of neighbors. This can lead to overfitting which should be avoided.
Decision Tree Classifier: As we can see, as the maximum depth increases the mean CV score also increases. An accuracy of 66.7% after hyper parameter tuning is good.
*Predicting:
Since we could conclude that the accuracy for KNN is much higher than Decision Tree, predicting the values with KNN is much more efficient. But however we will predict using both to compare both the scores.
Looking at the F1-scores, we can see the model is not very good when it comes to label 1,4,5. While it is doing extremely well for the other labels.
The F1 score for DT in label 4, 5 has a lower F1 score. Whilst the model is performing fairly well when looking at the other labels.
