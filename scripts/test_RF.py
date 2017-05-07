# Execution: ./bin/spark-submit --master local[8] --driver-memory 12g --executor-memory 12g ~/Desktop/spark_lib/test_RF.py > ~/Desktop/spark_lib/output/test_RF.op



from __future__ import print_function
import os
import shutil

from pyspark import SparkContext
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.regression import LabeledPoint
from numpy import array

# DEBUG environment
DEBUG = 0

# Parameters
# train_file = "/home/ajit/Desktop/spark_lib/dataset/cifar/cifar.data.train"
# test_file = "/home/ajit/Desktop/spark_lib/dataset/cifar/cifar.data.test"
train_file = "/home/ajit/Desktop/spark_lib/intermediate_output/cifar.ens.train"
test_file = "/home/ajit/Desktop/spark_lib/intermediate_output/cifar.ens.test"
num_class = 10
num_rf_trees = 25

# Configure the environment                                                     
if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = '$SPARK_HOME'

# Load and parse data
def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])

def appendColumn(ensemble_data, label_data):
	for i in range(0,len(ensemble_data),1):
		ensemble_data[i].extend([label_data[i]])


def writeToFile(en_data, filename):
	f = open(filename,'w+')
	f.writelines(' '.join(str(j) for j in i) + '\n' for i in en_data)
	f.close()


sc = SparkContext(appName="Ensemble_Classifier_CIFAR_Dataset")

data = sc.textFile(train_file)
train_data = data.map(parsePoint)

data = sc.textFile(test_file)
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


# Feature Data for Random Forest
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


# Read testfile
f = open('/home/ajit/Desktop/spark_lib/dataset/cifar/cifar.data.test')
r_test_features = []
l = f.readline()

while(l!=''):
    a = l.split()
    l2 = []
    for j in range(0,len(a),1):
        l2.extend([float(a[j])])
    r_test_features.append(l2)
    l = f.readline()

f.close()


# Build the Model
model = RandomForest.trainClassifier(train_data, num_class, {}, num_rf_trees)

rf_test_predict_label = []

# Predict Labels
for j in range(0, len(test_features), 1):
	p_l = model.predict(test_features[j])
	rf_test_predict_label.extend([p_l])

# Calculate Accuracy
sum_err = 0
for i in range(0, len(test_labels), 1):
	if(r_test_features[i][0]!=rf_test_predict_label[i]):
		print(str(r_test_features[i][0]) +' = '+ str(rf_test_predict_label[i]))
		sum_err = sum_err + 1

print(sum_err)
Err = sum_err/len(test_labels)
print("RF Classifier Error: " + str(Err))

sc.stop() 