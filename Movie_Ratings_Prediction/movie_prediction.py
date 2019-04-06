from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator


ratings_df_schema = StructType(
  [StructField('userId', IntegerType()),
   StructField('movieId', IntegerType()),
   StructField('rating', DoubleType())]
)
movies_df_schema = StructType(
  [StructField('ID', IntegerType()),
   StructField('title', StringType())]
)

spark = SparkSession.builder.appName('ALS Movie Predictions').getOrCreate()

movies_raw_df = spark.read.format("csv").option("header", "true").load("/home/ragesh/Data/Movie_Ratings/movies.csv", schema=movies_df_schema)
ratings_raw_df = spark.read.format("csv").option("header", "true").load("/home/ragesh/Data/Movie_Ratings/ratings.csv", schema=ratings_df_schema)

movies_raw_df.cache()
ratings_raw_df.cache()

# Movies with Highest Average Ratings
movie_ids_with_avg_ratings_df = ratings_raw_df.groupBy('movieId').agg(F.count(ratings_raw_df.rating).alias("count"), F.avg(ratings_raw_df.rating).alias("average"))
# print('movie_ids_with_avg_ratings_df:')
# movie_ids_with_avg_ratings_df.show(3, truncate=False)

movie_names_with_avg_ratings_df = movie_ids_with_avg_ratings_df.join(movies_raw_df, movies_raw_df.ID == movie_ids_with_avg_ratings_df.movieId) \
                                              .select(movie_ids_with_avg_ratings_df['average'], movies_raw_df.title, movie_ids_with_avg_ratings_df['count'], movie_ids_with_avg_ratings_df.movieId)
# print('movie_names_with_avg_ratings_df:')
# movie_names_with_avg_ratings_df.show(3, truncate=False)


# Movies with Highest Average Ratings and at least 500 reviews
movies_with_500_ratings_or_more = movie_names_with_avg_ratings_df.filter(movie_names_with_avg_ratings_df['count'] >= 500)\
                                                                 .sort(movie_names_with_avg_ratings_df['average'].desc())
# print('Movies with highest ratings:')
# movies_with_500_ratings_or_more.show(20, truncate=False)


# Split the dataset as training, validation and test
seed = 1800009193
(split_60_df, split_a_20_df, split_b_20_df) = ratings_raw_df.randomSplit([0.6, 0.2, 0.2], seed=seed)

# Let's cache these datasets for performance
training_df = split_60_df.cache()
validation_df = split_a_20_df.cache()
test_df = split_b_20_df.cache()


# Implement Alternating Least Squares
# Initialize our ALS learner
als = ALS()

# Set the parameters for the method
als.setMaxIter(5)\
   .setSeed(seed)\
   .setRegParam(0.1)\
   .setItemCol("movieId")\
   .setUserCol("userId")\
   .setRatingCol("rating")\

# Compute an evaluation metric for test dataset
# Create an RMSE evaluator using the label and predicted columns
reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="rating", metricName="rmse")

tolerance = 0.03
ranks = [4, 8, 12]
errors = [0, 0, 0]
models = [0, 0, 0]
err = 0
min_error = float('inf')
best_rank = -1

"""for rank in ranks:
    # Set the rank here:
    als.setRank(rank)
    # Create the model with these parameters.
    model = als.fit(training_df)
    # Run the model to create a prediction. Predict against the validation_df.
    predict_df = model.transform(validation_df)

    # Remove NaN values from prediction (due to SPARK-14489)
    predicted_ratings_df = predict_df.filter(predict_df.prediction != float('nan'))

    # Run the previously created RMSE evaluator, reg_eval, on the predicted_ratings_df DataFrame
    error = reg_eval.evaluate(predicted_ratings_df)
    errors[err] = error
    models[err] = model
    print('For rank %s the RMSE is %s' % (rank, error))
    if error < min_error:
        min_error = error
        best_rank = err
    err += 1

"""

als.setRank(12)
my_model = als.fit(training_df)
predict_df = my_model.transform(validation_df)
predicted_ratings_df = predict_df.filter(predict_df.prediction != float('nan'))
error = reg_eval.evaluate(predicted_ratings_df)
#print('The best model was trained with rank %s' % ranks[best_rank])
#my_model = models[best_rank]

# Run the best model with test dataset
predict_test_df = my_model.transform(test_df)

# Remove NaN values from prediction (due to SPARK-14489)
predicted_test_df = predict_df.filter(predict_test_df.prediction != float('nan'))

# Run the previously created RMSE evaluator, reg_eval, on the predicted_test_df DataFrame
test_RMSE = reg_eval.evaluate(predicted_test_df)

# Compute the average rating
avg_rating_df = training_df.select(F.avg(training_df.rating).alias("average"))

# Extract the average rating value. (This is row 0, column 0.)
training_avg_rating = avg_rating_df.collect()[0][0]

print('The average rating for movies in the training set is {0}'.format(training_avg_rating))

# Add a column with the average rating
test_for_avg_df = test_df.withColumn('prediction', F.lit(training_avg_rating))

# Run the previously created RMSE evaluator, reg_eval, on the test_for_avg_df DataFrame
test_avg_RMSE = reg_eval.evaluate(test_for_avg_df)












