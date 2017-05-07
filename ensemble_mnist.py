# Execution: ./bin/spark-submit --master local[8] --driver-memory 12g --executor-memory 12g ~/Desktop/spark_lib/ensemble_mnist.py > ~/Desktop/spark_lib/output/mnist.op



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
train_file = "/home/ajit/Desktop/spark_lib/dataset/mnist/mnist.train"
test_file = "/home/ajit/Desktop/spark_lib/dataset/mnist/mnist.test"
ensemble_train_file = "/home/ajit/Desktop/spark_lib/intermediate_output/mnist.ens.train"
ensemble_test_file = "/home/ajit/Desktop/spark_lib/intermediate_output/mnist.ens.test"

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


sc = SparkContext(appName="Ensemble_Classifier_MNIST_Dataset")

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




# C1 
# Build the Model
model = LogisticRegressionWithLBFGS.train(train_data, intercept=True, validateData=True, numClasses=10)

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
model = LogisticRegressionWithLBFGS.train(train_data, regParam=0.1, intercept=True, validateData=True, numClasses=10)

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
model = LogisticRegressionWithLBFGS.train(train_data, regParam=0.001, intercept=True, validateData=True, numClasses=10)

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
model = LogisticRegressionWithLBFGS.train(train_data, regParam=1, regType='l1', intercept=True, validateData=True, numClasses=10)

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
model = LogisticRegressionWithLBFGS.train(train_data, regParam=0.1, regType='l1', intercept=True, validateData=True, numClasses=10)

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
model = LogisticRegressionWithLBFGS.train(train_data, regParam=10, regType='none', intercept=True, validateData=True, numClasses=10)

# Predict Labels
c1_predict_labels_test_rdd = test_data.map(lambda p: ( model.predict(p.features)))
c1_predict_labels_train_rdd = train_data.map(lambda p: ( model.predict(p.features)))
c1_predict_labels_test = c1_predict_labels_test_rdd.collect()
c1_predict_labels_train = c1_predict_labels_train_rdd.collect()

# Append Labels
appendColumn(ensemble_test, c1_predict_labels_test)
appendColumn(ensemble_train, c1_predict_labels_train)


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


# Random Forest
# C13 - C21
# Build the Model
rf_trees = [25, 50, 100, 150, 200, 250, 300, 400]

for i in range(0, len(rf_trees), 1):
	num_rf_trees = rf_trees[i]

	# Build the Model
	model = RandomForest.trainClassifier(train_data, 10, {}, num_rf_trees)

	rf_train_predict_label = []
	rf_test_predict_label = []

	# Predict Labels
	for j in range(0, len(test_features), 1):
		p_l = model.predict(test_features[j])
		rf_test_predict_label.extend([p_l])

	for j in range(0, len(train_features), 1):
		p_l = model.predict(train_features[j])
		rf_train_predict_label.extend([p_l])


	# Append Labels
	appendColumn(ensemble_test, rf_test_predict_label)
	appendColumn(ensemble_train, rf_train_predict_label)


# Decision Trees
# C13 - C21
# Build the Model
max_depth = [5, 10, 15, 20]

for i in range(0, len(max_depth), 1):
	m_depth = max_depth[i]

	# Build the Model
	model = DecisionTree.trainClassifier(train_data, 10, {}, impurity='gini', maxDepth=m_depth)

	rf_train_predict_label = []
	rf_test_predict_label = []

	# Predict Labels
	for j in range(0, len(test_features), 1):
		p_l = model.predict(test_features[j])
		rf_test_predict_label.extend([p_l])

	for j in range(0, len(train_features), 1):
		p_l = model.predict(train_features[j])
		rf_train_predict_label.extend([p_l])


	# Append Labels
	appendColumn(ensemble_test, rf_test_predict_label)
	appendColumn(ensemble_train, rf_train_predict_label)

#----------
# Save Intermediate Ensemble Output
writeToFile(ensemble_train, ensemble_train_file)
writeToFile(ensemble_test, ensemble_test_file)


# Run Logistic Regression on intermediate ensemble data
data = sc.textFile(ensemble_train_file)
ens_train_data = data.map(parsePoint)

data = sc.textFile(ensemble_test_file)
ens_test_data = data.map(parsePoint)

# Build the Model & Predic labels for test data
model = LogisticRegressionWithLBFGS.train(ens_train_data, regParam=0.01, intercept=True, validateData=True, numClasses=10)
labelsAndPreds = ens_test_data.map(lambda p: (p.label, model.predict(p.features)))
testErr = labelsAndPreds.filter(lambda (v,p): v!=p).count()/float(ens_test_data.count())
print("Ensemble Classifier Error: " + str(testErr))

sc.stop() 
