# Execution: ./bin/spark-submit --master local[8] --driver-memory 12g --executor-memory 12g ~/Desktop/spark_lib/scripts/RF_BT.py > ~/Desktop/spark_lib/scripts/RF_BT.op



from __future__ import print_function
import os
import shutil

from pyspark import SparkContext
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.tree import GradientBoostedTrees
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from numpy import array

# DEBUG environment
DEBUG = 0

# Parameters
train_file = "/home/ajit/Desktop/spark_lib/dataset/synthetic/synth_data.train"
test_file = "/home/ajit/Desktop/spark_lib/dataset/synthetic/synth_data.test"

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


sc = SparkContext(appName="Ensemble_Classifier_Syntheti_cDataset")

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


# # Random Forest
# # C14 - C21
# # Build the Model
# rf_trees = 10


# num_rf_trees = rf_trees

# # Build the Model
# model = RandomForest.trainClassifier(train_data, 2, {}, num_rf_trees)

# rf_train_predict_label = []
# rf_test_predict_label = []

# # Predict Labels
# for j in range(0, len(test_features), 1):
# 	p_l = model.predict(test_features[j])
# 	rf_test_predict_label.extend([p_l])

# final_predict_labels = rf_test_predict_label


# # Gradient Boosted Trees
# # C14 - C21
# # Build the Model
# # Build the Model
model = GradientBoostedTrees.trainClassifier(train_data, {}, numIterations=10)

gbt_train_predict_label = []
gbt_test_predict_label = []

# Predict Labels
for j in range(0, len(test_features), 1):
	p_l = model.predict(test_features[j])
	gbt_test_predict_label.extend([p_l])
final_predict_labels = gbt_test_predict_label

# Calculate Precision, Recall and F1 Score

tp=0.0
fp=0.0
tn=0.0
fn=0.0

for i in range(0,len(test_labels),1):
    if( test_labels[i]==1 and final_predict_labels[i]==1 ):
        tp = tp+1
    elif( test_labels[i]==1 and (final_predict_labels[i]==0 or final_predict_labels[i]==-1) ):
        fn = fn+1
    elif( (test_labels[i]==0 or test_labels[i]==-1) and final_predict_labels[i]==1  ):
        fp = fp+1
    elif( (test_labels[i]==0 or test_labels[i]==-1) and (final_predict_labels[i]==0 or final_predict_labels[i]==-1) ):
        tn = tn+1

print("True Positive (TP): %.0f" % tp)
print("False Positive (FP): %.0f" % fp)
print("True Negative (TN): %.0f" % tn)
print("False Negative (FN): %.0f" % fn)

#  Precision and Recall

# Error
err = (fp+fn)/(tp+fp+tn+fn)
print("Error: %.3f" % err)

# Precision
p = tp/(tp+fp)
print("Precision: %.3f" % p)

# Recall
r = tp/(tp+fn)
print("Recall: %.3f" % r)

# Calculate F1 Score
f1 = 2*tp/(2*tp+fp+fn)
print("F1 Score: %.3f" % f1)



sc.stop() 
