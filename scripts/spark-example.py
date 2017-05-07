import os
from pyspark import SparkConf, SparkContext

# Configure the environment                                                     
if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = '$SPARK_HOME'

conf = SparkConf().setAppName('pubmed_open_access').setMaster('local[32]')
sc = SparkContext(conf=conf)

if __name__ == '__main__':
    ls = range(100)
    ls_rdd = sc.parallelize(ls, numSlices=1000)
    ls_out = ls_rdd.map(lambda x: x+1).collect()
    print 'output!: ', ls_out

## V- IMP (Command to execute) ------
# First cd $SPARK_HOME
# ./bin/spark-submit --master local[8] --driver-memory 12g --executor-memory 12g ~/Desktop/spark-example.py > ~/Desktop/output
