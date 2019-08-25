from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import functions as f


spark = SparkSession.builder.appName("Json writer").master("local").getOrCreate()

movies_schema = StructType([StructField("cast", ArrayType(StringType(), True), True),
                            StructField('genres', ArrayType(StringType(), True), True),
                            StructField("title", StringType(), True),
                            StructField("year", IntegerType(), True)
                            ])

raw_df = spark.read.json("file:///home/ragesh/Downloads/movies.json", schema=movies_schema)

raw_df.show(5, False)

