---
title: Conclusions
layout: default
nav_order: 4
---

## Approach

Our analysis incorporated a multitude of models ranging from logistic regressions to random forrests, each with their own underlying algorithms. All models were used to predict the classification of songs as either in the given Spotify playlist of not. Additionally, we experimented with model enhancing practices such as bagging and boosting to try and increase a given model's accuracy. 

## Results

The follow is a table that summarizes the training and test set accuracy for all the models we ran:


|                 Model Type                 | Training Accuracy   | Testing Accuracy   |
|--------------------------------------------|:-------------------:|:------------------:|
|        Baseline Logistic Regression        |       69.4%         |       67.1%        |
|  Logistic Regression With Quadratic Terms  |       49.7%         |       48.4%        |
|                     kNN                    |       63.1%         |       65.9%        |
| Logistic Regression With L1 Regularization |       88.6%         |       88.7%        |
| Logistic Regression With L2 Regularization |       69.2%         |       66.9%        |
|                     LDA                    |       88.1%         |       88.4%        |
|                     QDA                    |       86.6%         |       86.7%        |
|          Decision Tree Classifier          |       88.0%         |       89.0%        |
|    Decision Tree Classifier With Bagging   |       93.6%         |       91.8%        |
|               Random Forrest               |       92.9%         |       92.0%        |
|       Boosting Decision Tree Classifier    |       95.4%         |       93.0%        |
|                 Neural Net                 |       TBD           |       TBD          |

From these results we can see that there are two tiers of performance. The lowest performing models include the: baseline logistic regression, regression with quadratic terms, kNN, and logistic regression with L2 regularization. With the exception of these four models all the other ones perfrom somewhere betwen 88% to 93% on their test set accuracy measures. The best performing model is constructed by boosting a decision tree classifier. The accuracy attained by this model is 93.0%.

## Future Work

