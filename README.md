# Fraudulent Job Postings Detection using Gaussian Processes

## Abstract

This project investigates the use of Gaussian Process Regression (GPR), Gaussian Process Classification (GPC), and clustering techniques to identify fraudulent job postings in online recruitment platforms. The study compares several machine learning models including Normal Gaussian Classification, XGB Gaussian Classification, Gaussian Regression, and Logistic Regression to determine the most effective methods for detecting fake job postings. The primary goal is to identify fraudulent job data using these techniques.

**Keywords**: Gaussian Process, Classification, Regression, Clustering

---

## Table of Contents

1. [Introduction](#introduction)
2. [Literature Survey](#literature-survey)
3. [Problem and Dataset](#problem-and-dataset)
4. [Methods](#methods)
   - 4.1 [Clustering](#clustering)
   - 4.2 [Gaussian Classification](#gaussian-classification)
   - 4.3 [Gaussian Regression](#gaussian-regression)
   - 4.4 [Logistic Regression](#logistic-regression)
5. [Experimental Setup](#experimental-setup)
6. [Results](#results)
7. [Social, Ethical, Legal, and Professional Considerations](#social-ethical-legal-and-professional-considerations)
8. [Conclusion](#conclusion)
9. [References](#references)

---

## Introduction

The rise of fake job postings is a major concern in the online job recruitment domain. Fraudulent job postings are often designed to scam job seekers by requesting upfront payment or collecting personal data under the guise of a legitimate job opportunity. This paper introduces an automated system to detect fake job postings by leveraging various Gaussian Process techniques, including classification and regression models. The system utilizes clustering to group data and applies Gaussian Process algorithms to predict fraudulent job data.

---

## Literature Survey

- **Data Entry Scams, Pyramid Marketing, and Envelope Stuffing**: Fraudulent job postings often involve data entry scams, pyramid schemes, and envelope stuffing, where victims are promised high returns in exchange for payment or other personal data [1].
- **Gaussian Process Models**: A Gaussian Process is a powerful probabilistic model often used for function approximation and regression. This paper highlights the use of Gaussian processes for classification and regression tasks [2], [3].
- **Application in Image Processing**: K-means clustering has been successfully applied in image processing and data mining tasks, leading to enhanced decision-making and operational efficiency [6].

---

## Problem and Dataset

### Problem

The primary focus of this study is to identify fraudulent job postings. Fake job openings are typically designed to collect money from job seekers or steal their personal information. Detecting such fraudulent postings is a challenging but crucial task. By using various Gaussian Process techniques, we aim to detect fraudulent job postings accurately.

### Dataset

The dataset used in this study is sourced from Kaggle, titled "Predicting Fraudulent Job Data." It includes multiple input variables and one output variable. The input variables are:

- Title
- Location
- Department
- Salary Range
- Company Profile
- Description
- Requirements

The output variable is:

- Fraudulent (Binary: 1 for fraudulent, 0 for non-fraudulent)

[Dataset Link](https://www.kaggle.com/kj82227390/predicting-fraudulent-job/data)

---

## Methods

### Clustering
Clustering is performed as an unsupervised learning technique, grouping similar data points together. K-means clustering is used to create clusters based on similarity, where `k` is the number of clusters. This method involves two steps:
1. **Expectation**: Assign each data point to the nearest centroid.
2. **Maximization**: Compute new centroids by calculating the mean of all points in each cluster.

### Gaussian Classification
Gaussian Process Classification (GPC) is used to perform probabilistic classification by applying a Gaussian Process as a distribution over functions. This method predicts the likelihood of a data point belonging to a particular class.

- **XGB Gaussian Classifier**: This model applies XGBoost (eXtreme Gradient Boosting) to improve the efficiency of Gaussian classification using decision trees and boosting techniques.

### Gaussian Regression
Gaussian Process Regression (GPR) is applied for regression tasks, where the model fits data using probability distributions and provides uncertainty measures over predictions. The implementation is done using `scikit-learn`.

### Logistic Regression
Logistic Regression is an extension of linear regression used for binary classification tasks. It predicts the probability of the outcome based on the input features. In this study, it is used for classification of fraudulent job postings.

---

## Experimental Setup

### Data Preprocessing
The dataset is first preprocessed by handling missing values using the `fillna()` method. Preprocessing ensures that the dataset is ready for training the models.

### Feature Selection
Feature selection is performed to reduce the number of input variables, retaining only the most important ones. Seven input variables (Title, Location, Department, Salary Range, Company Profile, Description, and Requirements) are selected along with the output variable (Fraudulent).

### Feature Extraction
Feature extraction is done using `CountVectorizer`, which converts textual data into numerical features based on the frequency of words. This helps in transforming unstructured data into a format suitable for machine learning models.

---

## Results

### Clustering
The initial clustering step was performed using K-means, and scatter plots were created to visualize the grouping of data points based on input variables.

### Classification Accuracy
- **Gaussian Process Classifier**: 99.27% accuracy
- **XGB Gaussian Classifier**: 99.27% accuracy

Both models showed exceptional performance in identifying fraudulent job postings.

### Regression Accuracy
- **Logistic Regression**: 99% accuracy
- **Gaussian Regression**: 82% accuracy

Logistic regression outperformed Gaussian regression in terms of accuracy, demonstrating its suitability for this classification task.

---

## Social, Ethical, Legal, and Professional Considerations

The detection of fraudulent job postings using machine learning can have a significant social impact by preventing scams and protecting job seekers. However, there are ethical concerns, such as privacy issues related to handling personal information. The legal implications of identifying and reporting fake job postings must also be considered to avoid false accusations or violations of privacy.

---

## Conclusion

This paper demonstrates the effectiveness of Gaussian Process techniques in detecting fraudulent job postings. K-means clustering, along with Gaussian and XGB classification, as well as regression models, were implemented and compared. The models show promising results, with high accuracy in identifying fake job data. Further work can focus on enhancing model performance and addressing the social and ethical implications of detecting fraud in online job platforms.

---

## References

1. [Online Fake Job Postings](https://www.flexjobs.com/blog/post/common-job-search-scams-how-to-protect-yourself-v2/)
2. [Gaussian Process Introduction](https://www.researchgate.net/publication/41781206_Gaussian_Processes_in_Machine_Learning)
3. [Gaussian Process Regression](https://ieeexplore.ieee.org/document/7729450)
4. [Gaussian Process Classification](https://ieeexplore.ieee.org/document/8455721)
5. [Melanoma Classification Using XGBClassifier](https://ieeexplore.ieee.org/document/9498424)
6. [Introduction to Clustering](https://realpython.com/k-means-clustering-python/)
7. [Method for Gaussian Classifier](https://machinelearningmastery.com/gaussian-processes-for-classification-with-python/)
8. [XGB Classifier](https://practicaldatascience.co.uk/machine-learning/how-to-create-a-classification-model-using-xgboost)
9. [Gaussian Regression](https://towardsdatascience.com/quick-start-to-gaussian-process-regression-36d838810319)
10. [Logistic Regression](https://medium.com/@rajwrita/logistic-regression-the-the-e8ed646e6a29)

