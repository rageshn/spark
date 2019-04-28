from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import functions as f

spark = SparkSession.builder.appName("Basic Consumer with Kafka").master("local").getOrCreate()

movies_data = spark.readStream.format("kafka")\
                    .option("kafka.bootstrap.servers", "localhost:9092")\
                    .option("subscribe", "movies-sample").load()

movies_schema = StructType([StructField("cast", ArrayType(StringType(), True), True),
                            StructField("year", IntegerType(), True),
                            StructField("genres", ArrayType(StringType(), True), True),
                            StructField("title", StringType(), True)])

movies_value = movies_data.selectExpr("CAST(value as string)")
data = movies_value.select(f.from_json(f.col("value").cast("string"), movies_schema).alias("data")).select("data.*")
final = data.writeStream.outputMode("append").format("console").start()

final.awaitTermination()
