from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Basic Consumer with Kafka").master("local").getOrCreate()

movies_data = spark.readStream.format("kafka")\
                    .option("kafka.bootstrap.servers", "localhost:9092")\
                    .option("subscribe", "movies-sample").load()

movies_data.writeStream.outputMode("append").format("console").option("truncate", False).start().awaitTermination()
