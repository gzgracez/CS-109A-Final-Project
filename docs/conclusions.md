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
We generated a dataset by consolidating a large array of songs that vary in genre, language, tempo, rhythm, etc. We tried to curate a dataset that mimicked the variety of songs that Spotify has. Grace then had to go through these songs and classify whether she would like them in her playlist or not. Due to a multitude of constraints, we only had 5000 songs between both and training and test data. Ideally more songs that accurately capture the variety of songs that Spotify has would improve the training procedures for models.

# Adapting Playlists 
This entire project was built off of the preferences of one individual - Grace. While this proved to be a good proof of concept, future exploration should be done to analyze how the best model can help create playlists for others based upon their interests. 

# Collaborative Filtering
Finally collaborative filtering is another type of data modeling that is commonly used for recommendation algorithms. It is based on the fundamental idea that people perfer things similar to the things they've established they like. As such, it would be a good model to further investigate for this given project. 

# Improve Neural Network
There are many hyperparameters that can be tuned when configuring neural networks. Additionally, the number of epochs they run for along with the number of predictors are all factors that influence the accuracy and effectiveness of these networks. As such playing around with these variables would improve the model.
