"""

This is a spark batch job which consumes the ROSBag data in a distributed way and put the messages to kafka

"""

import sys
from functools import partial
import pandas as pd
import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession


# Use of msg_map to apply a function on all messages
def msg_map(r, func=str, conn={}):
    from collections import namedtuple
    from rosbag.bag import _get_message_type

    if r[1]['header']['op'] == 2 and r[1]['header']['conn'] == conn['header']['conn']:
        c = conn['data']
        c['datatype'] = str(c['type'])
        c['msg_def'] = str(c['message_definition'])
        c['md5sum'] = str(c['md5sum'])
        c = namedtuple('GenericDict', c.keys())(**c)
        msg_type = _get_message_type(c)
        msg = msg_type()
        msg.deserialize(r[1]['data'])
        yield func(msg)


# Create Sparkconf and load all the rosbag dependencies jar

sparkConf = SparkConf()
sparkConf.setAppName("ros_hadoop")
spark = SparkSession.builder.config(conf=sparkConf).getOrCreate()
sc = spark.sparkContext

# Read the ROSBag file using RosbagMapInputFormat with  index file
fin = sc.newAPIHadoopFile(
    path="/HMBfiles/HMB_3.bag",
    inputFormatClass="de.valtech.foss.RosbagMapInputFormat",
    keyClass="org.apache.hadoop.io.LongWritable",
    valueClass="org.apache.hadoop.io.MapWritable",
    conf={"RosbagInputFormat.chunkIdx": "HMB_3.bag.idx.bin"})

conn_a = fin.filter(lambda r: r[1]['header']['op'] == 7).map(lambda r: r[1]).collect()
conn_d = {str(k['header']['topic']): k for k in conn_a}

# see all topic names in rosbag
# print conn_d.keys()


# Take a messages from '/imu/data' topic using default str func
rdd = fin.flatMap(
    partial(msg_map, conn=conn_d['/imu/data'])
)

# Sample ROSBag data
# print(rdd.take(1)[0])


# using dataframe ROSBag data is mapped and converted into JSON and finally published into Kafka

from pyspark.sql import types as T
import yaml

schema = T.StructType()
schema = schema.add(T.StructField('seq', T.IntegerType()))
schema = schema.add(T.StructField('secs', T.IntegerType()))
schema = schema.add(T.StructField('nsecs', T.IntegerType()))
schema = schema.add(T.StructField('orientation_x', T.DoubleType()))
schema = schema.add(T.StructField('orientation_y', T.DoubleType()))
schema = schema.add(T.StructField('orientation_z', T.DoubleType()))
schema = schema.add(T.StructField('angular_velocity_x', T.DoubleType()))
schema = schema.add(T.StructField('angular_velocity_y', T.DoubleType()))
schema = schema.add(T.StructField('angular_velocity_z', T.DoubleType()))
schema = schema.add(T.StructField('linear_acceleration_x', T.DoubleType()))
schema = schema.add(T.StructField('linear_acceleration_y', T.DoubleType()))
schema = schema.add(T.StructField('linear_acceleration_z', T.DoubleType()))


def get_time_and_acc(r):
    r = yaml.load(r)

    return (r['header']['seq'],
            r['header']['stamp']['secs'],
            r['header']['stamp']['nsecs'],
            r['orientation']['x'],
            r['orientation']['y'],
            r['orientation']['z'],
            r['angular_velocity']['x'],
            r['angular_velocity']['y'],
            r['angular_velocity']['z'],
            r['linear_acceleration']['x'],
            r['linear_acceleration']['y'],
            r['linear_acceleration']['z'],
            )


# Create the dataframe from the topic "/imu/data"

pdf_acc = spark.createDataFrame(fin.flatMap(partial(msg_map, conn=conn_d['/imu/data']))
                                .map(get_time_and_acc), schema=schema)

# Save the messages into Kafka

pdf_acc.selectExpr("to_json(struct(*)) AS value").write.format("kafka")\
    .option("kafka.bootstrap.servers", "adasdemo1:9092,adasdemo2:9092,adasdemo3:9092").option("topic", "hello2").save()

print
"<<<<< Messages are successfully  published in to Kafka >>>>>"
spark.stop()
