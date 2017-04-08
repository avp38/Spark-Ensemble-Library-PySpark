# Execution: ./bin/spark-submit --master local[8] --driver-memory 12g --executor-memory 12g ~/Desktop/spark_lib/ensemble_breast_cancer.py > ~/Desktop/spark_lib/output/breast_cancer.op



from __future__ import print_function
import os
import shutil

from pyspark import SparkContext
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from numpy import array

# DEBUG environment
DEBUG = 0


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


sc = SparkContext(appName="Ensemble_Classifier_Breast_Cancer_Dataset")

data = sc.textFile("/home/ajit/Desktop/spark_lib/dataset/breast_cancer.data.train")
train_data = data.map(parsePoint)

data = sc.textFile("/home/ajit/Desktop/spark_lib/dataset/breast_cancer.data.test")
test_data = data.map(parsePoint)


# Debug data
#if DEBUG==1:
    #print(train_data.take(train_data.count()))

# Extract Labels
test_labels_rdd = test_data.map(lambda p: p.label)
test_labels = test_labels_rdd.collect()
ensemble_test = []
for i in range(0, len(test_labels),1):
	l1 = [test_labels[i]]
	ensemble_test.append(l1) 

train_labels_rdd = train_data.map(lambda p: p.label)
train_labels = train_labels_rdd.collect()
ensemble_train = []
for i in range(0, len(train_labels),1):
	l1 = [train_labels[i]]
	ensemble_train.append(l1) 


# C1 
# Build the Model
model = LogisticRegressionWithSGD.train(train_data)

# Predict Labels
c1_predict_labels_test_rdd = test_data.map(lambda p: ( model.predict(p.features)))
c1_predict_labels_train_rdd = train_data.map(lambda p: ( model.predict(p.features)))
c1_predict_labels_test = c1_predict_labels_test_rdd.collect()
c1_predict_labels_train = c1_predict_labels_train_rdd.collect()

# Append Labels
appendColumn(ensemble_test, c1_predict_labels_test)
appendColumn(ensemble_train, c1_predict_labels_train)


# C2 
# Build the Model
model = LogisticRegressionWithSGD.train(train_data, regParam=0.1)

# Predict Labels
c1_predict_labels_test_rdd = test_data.map(lambda p: ( model.predict(p.features)))
c1_predict_labels_train_rdd = train_data.map(lambda p: ( model.predict(p.features)))
c1_predict_labels_test = c1_predict_labels_test_rdd.collect()
c1_predict_labels_train = c1_predict_labels_train_rdd.collect()

# Append Labels
appendColumn(ensemble_test, c1_predict_labels_test)
appendColumn(ensemble_train, c1_predict_labels_train)


# C3 
# Build the Model
model = LogisticRegressionWithSGD.train(train_data, regParam=0.001)

# Predict Labels
c1_predict_labels_test_rdd = test_data.map(lambda p: ( model.predict(p.features)))
c1_predict_labels_train_rdd = train_data.map(lambda p: ( model.predict(p.features)))
c1_predict_labels_test = c1_predict_labels_test_rdd.collect()
c1_predict_labels_train = c1_predict_labels_train_rdd.collect()

# Append Labels
appendColumn(ensemble_test, c1_predict_labels_test)
appendColumn(ensemble_train, c1_predict_labels_train)


# C4 
# Build the Model
model = LogisticRegressionWithSGD.train(train_data, regParam=1, regType='l1')

# Predict Labels
c1_predict_labels_test_rdd = test_data.map(lambda p: ( model.predict(p.features)))
c1_predict_labels_train_rdd = train_data.map(lambda p: ( model.predict(p.features)))
c1_predict_labels_test = c1_predict_labels_test_rdd.collect()
c1_predict_labels_train = c1_predict_labels_train_rdd.collect()

# Append Labels
appendColumn(ensemble_test, c1_predict_labels_test)
appendColumn(ensemble_train, c1_predict_labels_train)


# C5 
# Build the Model
model = LogisticRegressionWithSGD.train(train_data, regParam=0.1, regType='l1')

# Predict Labels
c1_predict_labels_test_rdd = test_data.map(lambda p: ( model.predict(p.features)))
c1_predict_labels_train_rdd = train_data.map(lambda p: ( model.predict(p.features)))
c1_predict_labels_test = c1_predict_labels_test_rdd.collect()
c1_predict_labels_train = c1_predict_labels_train_rdd.collect()

# Append Labels
appendColumn(ensemble_test, c1_predict_labels_test)
appendColumn(ensemble_train, c1_predict_labels_train)


# C6 
# Build the Model
model = LogisticRegressionWithSGD.train(train_data, regParam=10, regType='none')

# Predict Labels
c1_predict_labels_test_rdd = test_data.map(lambda p: ( model.predict(p.features)))
c1_predict_labels_train_rdd = train_data.map(lambda p: ( model.predict(p.features)))
c1_predict_labels_test = c1_predict_labels_test_rdd.collect()
c1_predict_labels_train = c1_predict_labels_train_rdd.collect()



sc.stop() 
