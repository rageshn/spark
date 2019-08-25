from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import functions as f
from cassandra.cluster import Cluster


class ForeachWriter:

    cluster = None
    session = None

    def open(self, partition_id, epoch_id):
        #cluster = Cluster(["127.0.0.1"], port=9042)
        #session = cluster.connect(keyspace="movies", wait_for_all_pools=True)

        print("Partition ID: ", partition_id)
        print("Epoch/Batch ID: ", epoch_id)

    def process(self, row):
        print("Row: ", row)

    def close(self, error):
        print("Error: ", error)


spark = SparkSession.builder.appName("Basic Consumer with Kafka").getOrCreate()

movies_data = spark.readStream.format("kafka")\
                    .option("kafka.bootstrap.servers", "localhost:9092")\
                    .option("subscribe", "films").load()

"""movies_schema = StructType([StructField("id", StringType(), True, True),
                            StructField("directed_by", ArrayType(StringType(), True), True),
                            StructField("initial_release_date", DateType(), True),
                            StructField("name", StringType(), True),
                            StructField('genre', ArrayType(StringType(), True), True)])

movies_value = movies_data.selectExpr("CAST(value as string)")
#data = movies_value.select(f.from_json(f.col("value").cast("string"), movies_schema).alias("data")).select("data.*")
#data = movies_value.select(f.col("value").cast("string").alias("data")).select("data")
final = movies_value.writeStream.outputMode("append").format("console").start()

final.awaitTermination()
"""


movies_schema = StructType([StructField("cast", ArrayType(StringType(), True), True),
                            StructField('genres', ArrayType(StringType(), True), True),
                            StructField("title", StringType(), True),
                            StructField("year", IntegerType(), True)
                            ])

movies_data.printSchema()

movies_value = movies_data.selectExpr("CAST(value as string)")
data = movies_value.select(f.from_json(f.col("value").cast("string"), movies_schema).alias("data"))
final = data.writeStream.foreach(ForeachWriter()).start()

final.awaitTermination()
