# Execution: ./bin/spark-submit --master local[8] --driver-memory 12g --executor-memory 12g ~/Desktop/spark_lib/test_ensemble.py > ~/Desktop/spark_lib/output/test_ensemble.op



from __future__ import print_function
import os
import shutil

from pyspark import SparkContext
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from numpy import array


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


sc = SparkContext(appName="Ensemble_Classifier_Test")

data = sc.textFile("/home/ajit/Desktop/spark_lib/dataset/breast_cancer.data.train")
train_data = data.map(parsePoint)

data = sc.textFile("/home/ajit/Desktop/spark_lib/dataset/breast_cancer.data.test")
test_data = data.map(parsePoint)

# Build the Model
model = LogisticRegressionWithSGD.train(train_data, regParam=10, regType='none')


# Evaluating model on training data
labelsAndPreds = test_data.map(lambda p: (p.label, model.predict(p.features)))
testErr = labelsAndPreds.filter(lambda (v,p): v!=p).count()/float(test_data.count())
print("Error " + str(testErr))


sc.stop()