# Execution: ./bin/spark-submit --master local[8] --driver-memory 12g --executor-memory 12g ~/Desktop/spark_lib/scripts/svm.py > ~/Desktop/spark_lib/scripts/svm.op



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




# C1 
# Build the Model
model = LogisticRegressionWithSGD.train(train_data , regParam=1)




# SVM
# C7 
# Build the Model
#model = SVMWithSGD.train(train_data, regParam=1, regType='l2')




labelsAndPreds = test_data.map(lambda p: (p.label, model.predict(p.features)))
testErr = labelsAndPreds.filter(lambda (v,p): v!=p).count()/float(test_data.count())
print("Classifier Error: " + str(testErr))

final_predict_labels_rdd = test_data.map(lambda p: ( model.predict(p.features)))
final_predict_labels = final_predict_labels_rdd.collect()
# test_labels

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