"""

Reads and inserts the json data into cassandra table

"""

import json
from cassandra.cluster import Cluster

raw = open("/home/ragesh/Downloads/films.json", "r")

file_content = json.load(raw)
cluster = Cluster(["127.0.0.1"], port=9042)
session = cluster.connect(keyspace="movies", wait_for_all_pools=True)
i = 0

for val in file_content:
    i += 1
    col_names = []
    col_values = []
    for k, v in val.items():
        col_names.append(k)
        col_values.append(v)

    cols = "("
    values_to_insert = "("
    for c_n in col_names:
        cols += c_n + ","
        values_to_insert += "%s ,"
    cols = cols[:-1] + ")"
    values_to_insert = values_to_insert[:-1] + ")"

    session.execute("insert into movies_list " + cols + " values " + values_to_insert, tuple(col_values))

print("Rows inserted: ", i)

session.shutdown()
cluster.shutdown()
