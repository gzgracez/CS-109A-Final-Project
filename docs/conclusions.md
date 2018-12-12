---
title: Conclusions
layout: default
nav_order: 4
---

# Results

The following table summarizes the training and test set accuracy for all our models:


|                 Model Type                 | Training Accuracy   | Testing Accuracy   |
|--------------------------------------------|:-------------------:|:------------------:|
|        Baseline Logistic Regression        |       69.4%         |       67.1%        |
|  Logistic Regression With Quadratic Terms  |       49.7%         |       48.4%        |
| Logistic Regression With L1 Regularization |       88.6%         |       88.7%        |
| Logistic Regression With L2 Regularization |       69.2%         |       66.9%        |
|                     kNN                    |       63.1%         |       65.9%        |
|                     LDA                    |       88.1%         |       88.4%        |
|                     QDA                    |       86.6%         |       86.7%        |
|          Decision Tree Classifier          |       88.0%         |       89.0%        |
|               Random Forest                |       92.9%         |       92.0%        |
|    Decision Tree Classifier With Bagging   |       93.6%         |       91.8%        |
|       Boosting Decision Tree Classifier    |       95.4%         |       93.0%        |
|                 Neural Network             |       50.4%         |       40.4%        |

Our lowest performing models include the neural network, the logistic regression with quadratic terms, and our kNN model, all of which perform worse than the baseline.
We are not sure why the neural network.  
From these results we can see that there are two tiers of performance. 
The lowest performing models include the baseline logistic regression, regression with quadratic terms, kNN, and logistic regression with L2 regularization. 
With the exception of these four models all the other ones perfrom somewhere betwen 88% to 93% on their test set accuracy measures. 
The best performing model is constructed by boosting a decision tree classifier. 
The accuracy attained by this model is 93.0%.

## Future Work
# Data Inclusion
We generated a dataset of songs for Grace to classify by downloading her "favorites" and "unfavorites" playlists on Spotify. 
In the future, increasing the size of the dataset will be useful.
# Collaborative Filtering
# Improve Neural Network
# Other Playlists
In the future, we can repeat this analysis on the playlists of others to find the best performing model.