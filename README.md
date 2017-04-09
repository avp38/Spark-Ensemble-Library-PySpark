# Spark Ensemble Classifier (PySpark)

Ensemble Learning using Spark Framework (PySpark)





● Step 1: Learn first-level classifiers based on the original training data set. We have
several choices to learn base classifiers: 
1) We can apply Bootstrap sampling technique tolearn independent classifiers; 
2) we can adopt the strategy used in Boosting, i.e.,adaptively learn base classifiers based on data with a weight distribution; 
3) we can tune parameters in a learning algorithm to generate diverse base classifiers (homogeneous classifiers); 
4) we can apply different classification methods and/or sampling methods to generate base classifiers (heterogeneous classifiers).

● Step 2: Construct a new data set based on the output of base classifiers. Here the output predicted labels of the first-level classifiers are regarded as new features, and the original class labels are kept as the labels in the new data set. Instead of using predicted labels we could use probability estimations of first-level classifiers. We could also use different activation functions like Relu, Logistic or Tanh to create new features.

● Step 3 : Learn a second-level classifier based on the newly constructed data set. Any learning method could be applied to learn second-level classifier.

The Stacking is a general framework. We can plug in different classifiers and learning approaches to create the first-level features and transform the data into another feature space.