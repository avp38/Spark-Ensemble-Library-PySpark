# Execution:
# ./bin/spark-submit --master local[8] --driver-memory 12g --executor-memory 12g ~/Desktop/spark_lib/naive_bayes_example.py > ~/Desktop/spark_lib/output-2



from __future__ import print_function
import os
import shutil

from pyspark import SparkContext
# $example on$
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.util import MLUtils
import numpy as np
from pyspark.mllib.linalg import Vectors

# Configure the environment                                                     
if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = '$SPARK_HOME'

# $example off$

if __name__ == "__main__":

    sc = SparkContext(appName="PythonNaiveBayesExample")

    # $example on$
    # Load and parse the data file.
    data = MLUtils.loadLibSVMFile(sc, "/home/ajit/Desktop/spark_lib/sample_libsvm_data.txt")
    print(type(data))
    print(data)

    # Split data approximately into training (60%) and test (40%)
    training, test = data.randomSplit([0.6, 0.4])

    # Train a naive Bayes model.
    model = NaiveBayes.train(training, 1.0)

    # Make prediction and test accuracy.
    predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
    accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / test.count()
    print(type(predictionAndLabel))
    print('model accuracy {}'.format(accuracy))
    print(predictionAndLabel.take(predictionAndLabel.count()))

    predictionAndLabel = test.map(lambda p: (model.predict(p.features)))

    pdd3 = predictionAndLabel.union(predictionAndLabel)
    print(pdd3.take(pdd3.count()))

    nrows = predictionAndLabel.count()

    rows = (pdd3.zipWithIndex().groupBy(lambda (x, i): i / nrows).mapValues(lambda vals: [x for (x, i) in sorted(vals, key=lambda (x, i): i)]))

    print(rows.take(rows.count()))


   
