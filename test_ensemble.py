# Execution: ./bin/spark-submit --master local[8] --driver-memory 12g --executor-memory 12g ~/Desktop/spark_lib/test_ensemble.py > ~/Desktop/spark_lib/output/test_ensemble.op



from __future__ import print_function
import os
import shutil

from pyspark import SparkContext
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.tree import RandomForest
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


# sc = SparkContext(appName="Ensemble_Classifier_Test")

# data = sc.textFile("/home/ajit/Desktop/spark_lib/dataset/breast_cancer.data.train")
# train_data = data.map(parsePoint)

# data = sc.textFile("/home/ajit/Desktop/spark_lib/dataset/breast_cancer.data.test")
# test_data = data.map(parsePoint)

# Build the Model
model = RandomForest.trainClassifier(train_data, 2, {}, 100)
temp = model.predict([13.4, 20.52, 88.64, 556.7, 0.1106, 0.1469, 0.1445, 0.08172, 0.2116, 0.07325, 0.3906, 0.9306, 3.093, 33.67, 0.00541, 0.02265, 0.03452, 0.01334, 0.01705, 0.004, 16.41, 29.66, 113.3, 844.4, 0.1574, 0.3856, 0.5106, 0.2051, 0.3585, 0.1109])
print(temp)
# Evaluating model on training data
# labelsAndPreds = test_data.map(lambda p: (p.label, model.predict(p.features)))
# print(labelsAndPreds.take(labelsAndPreds.count()))
# testErr = labelsAndPreds.filter(lambda (v,p): v!=p).count()/float(test_data.count())
# print("Error " + str(testErr))
# print(labelsAndPreds.take(labelsAndPreds.count()))

sc.stop()