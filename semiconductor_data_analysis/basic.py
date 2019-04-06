from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local").appName("SemiConductor Analysis").getOrCreate()

train_df = spark.read.format("csv").load("/home/ragesh/Data/Semi_Conductor/train.csv", header="true", inferschema=True)
unique_m_df = spark.read.format("csv").load("/home/ragesh/Data/Semi_Conductor/unique_m.csv", header="true", inferschema=True)

print(unique_m_df.printSchema())