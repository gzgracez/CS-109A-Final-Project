---
title: Conclusions
layout: default
nav_order: 4
---

# Results
Our best model performs with an accuracy of 95.4% in the training set, and 93.0% in the test set.
The following table summarizes the accuracies for all our models, ordered by accuracy in the test set:

|                 Model Type                 | Train Accuracy      | Test Accuracy      |
|--------------------------------------------|:-------------------:|:------------------:|
|  Logistic Regression With Quadratic Terms  |       49.7%         |       48.4%        |
|                 Neural Network             |       50.5%         |       51.6%        |
|                     kNN                    |       63.1%         |       65.9%        |
| Logistic Regression With L2 Regularization |       69.2%         |       66.9%        |
|        Baseline Logistic Regression        |       69.4%         |       67.1%        |
|                     QDA                    |       86.6%         |       86.7%        |
|                     LDA                    |       88.1%         |       88.4%        |
| Logistic Regression With L1 Regularization |       88.6%         |       88.7%        |
|          Decision Tree Classifier          |       88.0%         |       89.0%        |
|    Decision Tree Classifier With Bagging   |       93.6%         |       91.8%        |
|               Random Forest                |       92.9%         |       92.0%        |
|       Boosted Decision Tree Classifier     |       95.4%         |       93.0%        |

Our lowest performing models include the logistic regression with quadratic terms, the neural network, and the kNN model, all of which perform worse than the baseline.
Our best performing models were all ensemble methods with the decision tree, with the boosted decision tree classifier, the random forest model, and the decision tree classifier with bagging performing best.

## Future Work
# Data Inclusion
We generated a dataset of songs for Grace to classify by downloading her "favorites" and "unfavorites" playlists on Spotify. 
In the future, increasing the size of the dataset will be useful.
# Collaborative Filtering
# Improve Neural Network
# Other Playlists
In the future, we can repeat this analysis on the playlists of others to find the best performing model.