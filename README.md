Abstract- Main study of this paper is it deals with gaussian process regression, gaussian process classification and clustering. Two different regression and classification techniques are performed 1.e., normal gaussian classification and XGB gaussian classification, gaussian regression and logistic regression to check how efficiently the models are working by comparing them with outputs. Main goal is to find out the fraudulent data of the job postings by performing different techniques.
Keywords-gaussian process, classification, regression, clustering  
II.	INTRODUCTION 
Fake jobs are the one of the main concerns that should be highlighted in domain of online job recruitment. company owners will be posting all the job vacancies online so that people who are unemployed can get a posting very easily by choosing them.so, there will be few peoples where they act as company owners and post job vacancies online so that they can request money and cheat job seekers.so here an automated tool is developed so that we can easily find out the fake job postings. Different Gaussian process are being implemented using machine learning to find out these fake job postings.so initially we are performing the clustering of the data using k-means algorithm and then classification and regression are performed using the gaussian process algorithms.
III.	LITERATURE SURVEY
[1] Based on several studies it has shown that the main reasons for fake job postings are data entry scams, pyramid marketing, stuffing envelopes etc.

Data entry scams:
One of main reason for data entry scam is that they will show a lot of funding for the job where less skill is required. Here scam will take place of upfront funds.

Pyramid marketing:
This has no basis in real commerce and there will be only exchange of money no product will be involved. most of the people invest money in this marketing because people trust that they will be funded in huge amounts but the open reality is some one must lose money if the other one wants to gain.
Stuffing envelopes:
This is one of biggest online job scams that will taking       place currently. It is more like filling an envelope online and paying a small amount online. Making others to buy the same envelope online will get a small commission but others will be involved in scam and pay an amount that is not refundable.

[2] A basic introduction for the gaussian process models is given and this paper mainly focuses on distribution over functions using stochastic process. Training data will be incorporated using simple equations and will see how marginal likelihood will be used to learn hyperparameters. All the current trends and practical advantages in gaussian process will be explained here. 

[3] In this paper gaussian regression being used to estimate the tree height. In a coniferous boreal forest, the tree estimates are produced. Both predictive and estimate variance are found for gaussian process of each pixel. Results that are obtained shows that the estimates that are produced are really good and efficient when compared to the previous models. 

[4] The main aim here is to go on to either labelled or unlabelled data to train a mathematical model and then we can use to predict the unlabelled data for data mining model.  It is tough task to match the training datasets which has same no of distribution values. In this paper a Transfer classification algorithm has been proposed based on the gaussian process model, which is used to solve homogeneous transfer classification problem. 



[5] one of the most occurring cancer melanoma, patients moles will be evaluated individually to identify outlier lesions. In recent past there are high level results in medical field using machine learning and deep learning techniques. Skin lesion classification is performed by training a deep neural network on image dataset and it is classified effectively using XGB classifier, using this the accuracy will increase and computational costa are decreased.



[6] In image processing clustering is the initial task. In this paper database contains of data in form of images and texts. These databases should be mined continuously and decisions are made accurately in short span of time to increase profit in marketing. Image segmentation plays a crucial role in clustering. By using this technique efficiency will be increased and time will be saved. In this paper area of image segmentation deals with the application of standard k-means and fuzzy k-means algorithms. So finally various experiments have proven that the use of k-means algorithms has shown high efficiency in image segmentation.
		

IV.	PROBLEM AND DATA SET

Problem:
We are mainly dealing with the fake job postings that are being posted online. Numerous company owners make fake job openings online so that they can attract the people in name of high fundings and they can add their resume in their file and add them falsely as replacement of staff .so to find out this fake job posting we are going to using different gaussian process techniques to predict the percentage of these fake job postings.


Dataset:

We are using a dataset that is taken from Kaggle website known as “predicting fraudulent job-data. There are multiple input variables in this dataset and a output variable. we are choosing seven input variables and one output variable. the seven input variables are title, location, department, salary_range, company_profile, description, requirements and the output variable are fraudulent. By using this data, we are going to perform the gaussian process techniques.
Link for the dataset is given below:
https://www.kaggle.com/kj82227390/predicting-fraudulent-job/data.

V.	METHODS
Methods that are performed in this experiment are shown below: 
1)clustering 
2)Gaussian Classification:
In this again we are performing the two types of classification techniques one is normal gaussian classification and other XGB gaussian classifier 
3)Gaussian regression
4)logistic regression


1)clustering:
	It is an unsupervised machine learning task, where all the unlabeled data will be grouped together. All the data points that are similar will be formed into different clusters. K-means clustering:
	[6] The initial step in k-means clustering is that we are selecting k centroids, where k is no of cluster that have been chosen. Center of the cluster is represented by data points which are known as centroids. K-means mainly works in two step process:
(1) expectation
(2) maximization
(1) expectation is defined as nearest centroid chosen by assigning each data point
(2) maximization is defined as where new centroid will be set by computing all the mean points for each cluster

2)Gaussian classification:
	[7] To perform probilistic classification we are using the gaussian classifier. A gaussian process is defined as a probability distribution generalization, stochastic process that governs the properties of functions. we can import gaussian processor classifier class from sikit learn in python. Using this we can specify the kernel. Model will best fit the kernel for the training dataset once the kernel is specified.no of iterations for the optimizer is controlled by max_iter_predict.

XGB classifier:
	[8] XG boost uses a process know as boosting which is used to improve the model’s efficiency and it is based on the decision tree machine learning algorithm. Once the data has been imported using that dataset model will be trained.  Basic classification involves in predicting the target class. Model will try to learn the features that are correlated with target class and once this is done model will be more accurate at making predictions. XGboost has been implemented in various languages, in python we are using it by importing sklearn framework.


3)Gaussian regression:
	[9] Gaussian process regression is framework of supervised machine learning which is mostly used for the regression.  Using this process, we can provide uncertainty measures over predictions. Gaussian process regression fits all the data with the help of functions finds the probability distribution. There are many ways to implement the gaussian process but we are using scikitlearn to implement gaussian process regression 



4)logistic regression:
	[10] Linear regression extension is known as logistic regression. It shows two outcomes for the classification problems. This is mainly medical fields like to predict diseases early stages on the basis of age, problems etc. logistic regression is same as the linear regression just that it predicts probabilities for multiple external factors. In logistic regression the dependent variable follows the Bernoulli distribution. maximum likelihood estimation is used to calculate the estimation.
The accuracy for logistic regression is calculated by below formulae
(TP+TN)/(TP+TN+FP+FN)
Where TP=True Positive, TN=True Negative, FP=False Negative, FN=False Negative



VI.	EXPERIMENTAL SETUP
Data preprocessing:
	It is defined as the technique that is used to preparing the dataset for the training of our machine learning model. 
 Initially we are importing the dataset and then we filling all the null values using fillna method. 

 
Feature selection:
	Feature selection is defined as decreasing the no of input variables and taking selected variables into consideration to predict the target variable. The main focus here is to remove all the unwanted information from the dataset.
In the dataset totally there are 18 variables in that we are choosing eight variables in that seven input variables and one output variable they are title, location, department, salary_range, company_profile, description, requirements and the output variable is fraudulent.


Feature extraction:
Feature extraction is defined as a method which is used to combine the variables into features where the original dataset won’t be disturbed by doing this.  One the main thing about feature extraction is that all the redundant data will be removed very easily.
The feature extraction technique is being used here count vectorizer. It is one of best tool that has been provided by scikitlearn.  The conversion of words into vector on the basis of number of times that occurred in entire text. Screen shot after converting the dataset into is displayed below 


 

VII.	RESULTS

Clustering has been performed for the input data and output data 

 

The plot for clustering has been plotted below for all the input variables and output variable

 



 
	

Here scatter plot has been plotted between all the input variables against the output variable.
 
Once the classification techniques are performed the output graphs and accuracy score are displayed below 

 




 
	

The accuracy score for gaussian process classifier and XGB classifier are 99.27 percent respectively. This shows that both the models are working very accurately.

 


The above graph is a regression graph plot. The accuracy score for logistic regression is 99 percent and the accuracy score for Gaussian regression is 82 percent. This shows that 
Logistic regression model work more accurately that gaussian regression model.

VIII.	SOCIAL, ETHICAL, LEGAL AND PROFESSIONAL CONSIDERATIONS 
The dataset is mainly based on the finding out fake job postings. The dataset consists of data like job title, company name, location etc. people might post few fake postings online so that others ones can easily apply for the job and pays some upfront this may cause social issue and few other ones can speak to people online by telling them company may provide a job with high hike which mainly leads to ethical breaching and manager or employee who is breaching these rules is breaking the company professionality and may subjected to causes as mentioned on company’s privacy site. We have developed a model to find out the fake job postings. It is better to find out people who are making these fake job openings because the scams will be reduced.
IX.	CONCLUSION
This paper mainly discussed about the Gaussian process techniques. Initially clustering techniques has been done using k-means algorithm and graph has been plotted. Gaussian process regression and logistic regression, gaussian classification and XGB classification techniques are done and outputs have been compared and graphs are plotted respectively.
IX. REFERENCES
[1] “online fake job postings” https://www.flexjobs.com/blog/post/common-job-search-scams-how-to-protect-yourself-v2/
[2]” Gaussian process introduction” https://www.researchgate.net/publication/41781206_Gaussian_Processes_in_Machine_Learning 
[3]” Gaussian process regression” https://ieeexplore.ieee.org/document/7729450 
[4] “Gaussian process classification” https://ieeexplore.ieee.org/document/8455721 
[5]” MELANOMA CLASSIFICATION USING XGBCLASSIFIER AND EFFICIENTNET” HTTPS://IEEEXPLORE.IEEE.ORG/DOCUMENT/9498424 
 [6]”Introduction to clustering” https://realpython.com/k-means-clustering-python/ 
 [7]”method for gaussian classifier” https://machinelearningmastery.com/gaussian-processes-for-classification-with-python/ 
[8]”XGB classifier” https://practicaldatascience.co.uk/machine-learning/how-to-create-a-classification-model-using-xgboost 
[9]” Gaussian regression” https://towardsdatascience.com/quick-start-to-gaussian-process-regression-36d838810319 
[10]”logistic regression” https://medium.com/@rajwrita/logistic-regression-the-the-e8ed646e6a29
