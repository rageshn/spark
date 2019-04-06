from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import *


def main():
    filepath = "/home/ragesh/Data/Apache_Logs/access_log_Aug95"
    spark = SparkSession.builder.appName("log analysis").getOrCreate()
    raw_df = readandprocessfile(filepath)
    raw_df.show(10, truncate=False)


def readandprocessfile(filepath):
    spark = SparkSession.builder.getOrCreate()
    raw_df = spark.read.text(filepath)
    return raw_df


if __name__ == "__main__":
    main()



