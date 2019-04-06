# Apache Log processing
# http://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html

from pyspark.sql import SparkSession
from pyspark.sql.functions import split, regexp_extract
from pyspark.sql.functions import col, sum, lit, concat, udf
from pyspark.sql.functions import min, avg, max, desc, dayofmonth

spark = SparkSession.builder.appName('Apache Log Processing').master("local").getOrCreate()
# print(spark.version)

filePath = "/home/ragesh/Data/Apache_Logs/access_log_Aug95"

raw_df = spark.read.text(filePath)

# raw_df.show(10, truncate=False)
# raw_df.printSchema()

# Splitting the entry into columns
split_df = raw_df.select(regexp_extract('value', r'^([^\s]+\s)', 1).alias('host'),
                         regexp_extract('value', r'^.*\[(\d\d/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} -\d{4})]', 1).alias('timestamp'),
                         regexp_extract('value', r'^.*"\w+\s+([^\s]+)\s+HTTP.*"', 1).alias('path'),
                         regexp_extract('value', r'^.*"\s+([^\s]+)', 1).cast('integer').alias('status'),
                         regexp_extract('value', r'^.*\s+(\d+)$', 1).cast('integer').alias('content_size'))

# split_df.show(10, truncate=False)

# Checking for empty records
# print(raw_df.filter(raw_df['value'].isNull()).count())
# empty_rows_df = split_df.filter(split_df['host'].isNull() |
#                                split_df['timestamp'].isNull() |
#                                split_df['path'].isNull() |
#                                split_df['status'].isNull() |
#                                split_df['content_size'].isNull())

# print(empty_rows_df.count())

# Checks whether the column has null values or not
# def count_null(col_name):
#    return sum(col(col_name).isNull().cast('integer')).alias(col_name)


# exprs = []
# for col_name in split_df.columns:
#    exprs.append(count_null(col_name))

# split_df.agg(*exprs).show()

# bad_content_size_df = raw_df.filter(~raw_df['value'].rlike(r'\d+$'))
# print(bad_content_size_df.count())

# bad_content_size_df.select(concat(bad_content_size_df['value'], lit('*'))).show(truncate=False)

cleaned_df = split_df.fillna({'content_size': 0})

# exprs = []
# for col_name in cleaned_df.columns:
#    exprs.append(count_null(col_name))

# cleaned_df.agg(*exprs).show()

month_map = {
  'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7,
  'Aug': 8,  'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

def parse_clf_time(s):
    """ Convert Common Log time format into a Python datetime object
    Args:
        s (str): date and time in Apache time format [dd/mmm/yyyy:hh:mm:ss (+/-)zzzz]
    Returns:
        a string suitable for passing to CAST('timestamp')
    """
    # NOTE: We're ignoring time zone here. In a production application, you'd want to handle that.
    return "{0:04d}-{1:02d}-{2:02d} {3:02d}:{4:02d}:{5:02d}".format(
      int(s[7:11]),
      month_map[s[3:6]],
      int(s[0:2]),
      int(s[12:14]),
      int(s[15:17]),
      int(s[18:20])
    )


u_parse_time = udf(parse_clf_time)

logs_df = cleaned_df.select('*', u_parse_time(cleaned_df['timestamp']).cast('timestamp').alias('time')).drop('timestamp')

logs_df.cache()

# Content Size statistics
# -------------------------
content_size_summary_df = logs_df.describe(['content_size'])
# content_size_summary_df.show()

# Alternate method

content_size_stats = content_size_summary_df.agg(
    min(content_size_summary_df['content_size']),
    avg(content_size_summary_df['content_size']),
    max(content_size_summary_df['content_size'])
).first()

# print(content_size_stats[1])


# HTTP Status analysis
# --------------------
status_by_count_df = logs_df.groupby('status').count().sort('status')

# status_by_count_df.show()

# Frequent hosts
# --------------
host_sum_df = logs_df.groupBy("host").count()
# host_sum_df.show(10, truncate=False)
host_more_than_10_df = host_sum_df.filter(host_sum_df['count'] > 10).select(host_sum_df['host'])
# host_more_than_10_df.show(20, truncate=False)


# Frequent paths
# ---------------

# paths_df = logs_df.groupBy("path").count().sort('count', ascending=False)
# paths_df.show(10, truncate=False)
# paths_count_df = paths_df.select('path', 'count').rdd.map(lambda r: (r[0], r[1]))
# paths_count_df.collect()


# Counts by Status paths
# ----------------

# logs_df.show(10, truncate=False)

valid_req_df = logs_df.filter(logs_df['status'] != '200')
# valid_req_df.show(10, truncate=False)
valid_req_count_df = valid_req_df.groupBy('status').count().sort('count', ascending=False)
# valid_req_count_df.show(20, truncate=False)
top_paths = valid_req_df.groupBy('path').count().sort('count', ascending=False)
# top_paths.show(20, truncate=False)


# Top Error paths
# ----------------

error_df = logs_df.filter(logs_df['status'] == '200')
error_paths_df = error_df.groupBy('path').count().sort('count', ascending=False)
# error_paths_df.show(10, truncate=False)

# Unique Hosts Count
# -------------------

# unique_hosts_df = logs_df.dropDuplicates(['host']).count()
# print(unique_hosts_df)

# Unique host count by day
# -------------------------

# logs_df.show(10, truncate=False)
day_to_host_df = logs_df.select('host', dayofmonth('time').alias('day')).sort('day', ascending=False)
# day_to_host_df.show(10, truncate=False)
day_to_host_unique_df = day_to_host_df.dropDuplicates()
daily_hosts_df = day_to_host_unique_df.groupBy('day').count()
# daily_hosts_df.show(31, truncate=False)
daily_hosts_df.cache()

# Average requests per day
total_req_per_day_df = logs_df.groupBy(dayofmonth('time').alias('day')).count()
# total_req_per_day_df.show(31)

avg_re_per_day_df = total_req_per_day_df.join(daily_hosts_df, ['day']).\
    select('day', (total_req_per_day_df['count']/daily_hosts_df['count']).cast('integer').alias('Average_Requests'))

avg_re_per_day_df.show(31, truncate=False)