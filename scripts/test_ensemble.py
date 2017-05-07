# Execution: ./bin/spark-submit --master local[8] --driver-memory 12g --executor-memory 12g ~/Desktop/spark_lib/test_ensemble.py > ~/Desktop/spark_lib/output/test_ensemble.op



from __future__ import print_function
import os
import shutil

from pyspark import SparkContext
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.tree import GradientBoostedTrees
from pyspark.mllib.regression import LabeledPoint
from numpy import array
import numpy as np

# Configure the environment                                                     
if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = '$SPARK_HOME'

# Load and parse data
def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])

def appendColumn(ensemble_data, label_data):
	for i in range(0,len(ensemble_data),1):
		ensemble_data[i].extend(label_data[i])


# Parameters
train_file = "/home/ajit/Desktop/spark_lib/dataset/breast_cancer.data.train"
test_file = "/home/ajit/Desktop/spark_lib/dataset/breast_cancer.data.test"


sc = SparkContext(appName="Ensemble_Classifier_Test")


data = sc.textFile(train_file)
train_data = data.map(parsePoint)

data = sc.textFile(test_file)
test_data = data.map(parsePoint)


test_labels_rdd = test_data.map(lambda p: p.label)
test_labels = test_labels_rdd.collect()

# Feature Data for Random Forest and Boosted Trees (List of Lists)
# Read trainfile
f = open(train_file)
train_features = []
l = f.readline()

while(l!=''):
    a = l.split()
    l2 = []
    # Note: Started with index 1 to avoid the label
    for j in range(1,len(a),1):
        l2.extend([float(a[j])])
    train_features.append(l2)
    l = f.readline()

f.close()

# Read testfile
f = open(test_file)
test_features = []
l = f.readline()

while(l!=''):
    a = l.split()
    l2 = []
    for j in range(1,len(a),1):
        l2.extend([float(a[j])])
    test_features.append(l2)
    l = f.readline()

f.close()

# Build the Model
# num_rf_trees = 1
# model = RandomForest.trainClassifier(train_data, 2, {}, num_rf_trees, maxDepth=2)
model = GradientBoostedTrees.trainClassifier(train_data, {})
print(model)
print(model.toDebugString())
rf_test_predict_label = []

sum_error = 0
# Predict Labels
for j in range(0, len(test_features), 1):
	p_l = model.predict(test_features[j])
	rf_test_predict_label.extend([p_l])
	if p_l!=test_labels[j]:
		sum_error= sum_error+1
print(rf_test_predict_label)
print(test_labels)
accuracy = sum_error/len(test_features)
print("Error: " + str(accuracy))



# data = sc.textFile(train_file)
# train_data = data.map(parsePoint)

# data = sc.textFile(test_file)
# test_data = data.map(parsePoint)


# # Build the Model
# model = LogisticRegressionWithSGD.train(train_data, regParam=0.01, regType='l2')

# # model = RandomForest.trainClassifier(train_data, 2, {}, 100)
# # temp = model.predict([13.4, 20.52, 88.64, 556.7, 0.1106, 0.1469, 0.1445, 0.08172, 0.2116, 0.07325, 0.3906, 0.9306, 3.093, 33.67, 0.00541, 0.02265, 0.03452, 0.01334, 0.01705, 0.004, 16.41, 29.66, 113.3, 844.4, 0.1574, 0.3856, 0.5106, 0.2051, 0.3585, 0.1109])
# # print(temp)


# # Evaluating model on training data
# labelsAndPreds = test_data.map(lambda p: (p.label, model.predict(p.features)))
# print(labelsAndPreds.take(labelsAndPreds.count()))
# testErr = labelsAndPreds.filter(lambda (v,p): v!=p).count()/float(test_data.count())
# print("Error " + str(testErr))
# print(labelsAndPreds.take(labelsAndPreds.count()))

sc.stop()