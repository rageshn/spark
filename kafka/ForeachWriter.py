"""
Foreach writer class for writing data to cassandra
"""

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
        print(row)

    def close(self, error):
        print("Error: ", error)
