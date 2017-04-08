# Execution:
# ./bin/spark-submit --master local[8] --driver-memory 12g --executor-memory 12g ~/Desktop/spark_lib/example-2.py > ~/Desktop/spark_lib/output-2


import os
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

sc = SparkContext(appName="LogisticRegression")

data = sc.textFile("/home/ajit/Desktop/spark_lib/sample_svm_data.txt")
parsedData = data.map(parsePoint)

# Build the Model
model = LogisticRegressionWithSGD.train(parsedData)

# Evaluating model on training data
labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda (v,p): v!=p).count()/float(parsedData.count())
print("Training Error" + str(trainErr))

sc.stop()