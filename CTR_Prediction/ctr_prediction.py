"""
Part 1: Featurize categorical data using one-hot-encoding (OHE)

Part 2: Construct an OHE dictionary

Part 3: Parse CTR data and generate OHE features

Visualization 1: Feature frequency
Part 4: CTR prediction and logloss evaluation

Visualization 2: ROC curve
Part 5: Reduce feature dimension via feature hashing

"""

from pyspark.sql import SparkSession
from collections import defaultdict
import numpy as np
from pyspark.mllib.linalg import SparseVector

spark = SparkSession.builder.appName("Click Through Rate Prediction using OHE").getOrCreate()





