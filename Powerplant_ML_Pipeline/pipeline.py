from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

spark = SparkSession.builder.appName("Power Plant ML Pipeline").master("local").getOrCreate()

power_data_schema = StructType([
    StructField('Atmospheric_Temperature', DoubleType(), True),
    StructField('Vacuum_Speed', DoubleType(), True),
    StructField('Atmospheric_Pressure', DoubleType(), True),
    StructField('Relative_Humidity', DoubleType(), True),
    StructField('Power_Output', DoubleType(), True)
])

raw_data_df = spark.read.format("csv").option("delimiter", "\t").option("header", "true").\
    load("/home/ragesh/Data/Power_Plant_Data/power_plant_data", schema=power_data_schema)

# raw_data_df.show(10, truncate=False)

# Converts the list of columns into a single vector column
vectorizer = VectorAssembler()
vectorizer.setInputCols(['Atmospheric_Temperature', 'Vacuum_Speed', 'Atmospheric_Pressure', 'Relative_Humidity'])
vectorizer.setOutputCol('features')

# splitting the dataset into training and test datasets in 80% - 20% ratio
seed = 1800009193
(testSetDF, trainSetDF) = raw_data_df.randomSplit([0.2, 0.8], seed=seed)
testSetDF.cache()
trainSetDF.cache()

# Create a Linear Regression Model
lr = LinearRegression()
# print(lr.explainParams())
lr.setPredictionCol('Predicted_PE').setLabelCol('Power_Output').setMaxIter(100).setRegParam(0.1)

# Create a ML Pipeline and set the stages
lrPipeline = Pipeline()
lrPipeline.setStages([vectorizer, lr])

# Train the model with training dataset
lrModel = lrPipeline.fit(trainSetDF)

# Get the intercept and co-efficients of the equation
intercept = lrModel.stages[1].intercept
weights = lrModel.stages[1].coefficients

# Get list of column names except output
features = [col for col in trainSetDF.columns if col != "Power_Output"]

coefficents = zip(weights, features)

# sort the coefficients from greatest absolute weight most to the least absolute weight
coefficents = sorted(coefficents, key=lambda tup: abs(tup[0]), reverse=True)

# equation = "y = {intercept}".format(intercept=intercept)
# variables = []
# for x in coefficents:
#     weight = abs(x[0])
#    name = x[1]
#    symbol = "+" if (x[0] > 0) else "-"
#    equation += (" {} ({} * {})".format(symbol, weight, name))

# Final equation
# print("Linear Regression Equation: " + equation)

# Applying the LR model to test dataset
# predictionsAndLabelsDF = lrModel.transform(testSetDF).select("Atmospheric_Temperature", "Vacuum_Speed", "Atmospheric_Pressure", "Relative_Humidity", "Power_Output", "Predicted_PE")

# predictionsAndLabelsDF.show(10, truncate=False)

# Use Regression Evaluator to calculate RMSE
regEval = RegressionEvaluator(predictionCol="Predicted_PE", labelCol="Power_Output", metricName="rmse")
# rmse = regEval.evaluate(predictionsAndLabelsDF)
# print(rmse)

# Calculate co-efficient of determination (R-squared)
# r2 = regEval.evaluate(predictionsAndLabelsDF, {regEval.metricName: "r2"})
# print(r2)


# Create a 3 fold cross validation
crossVal = CrossValidator(estimator=lrPipeline, evaluator=regEval, numFolds=3)

# Regularization parameter from 0.01 to 0.10
regParam = [x / 100.0 for x in range(1, 11)]

# Create a parameter grid using the ParamGridBuilder, and add the grid to the CrossValidator
paramGrid = ParamGridBuilder().addGrid(lr.regParam, regParam).build()
crossVal.setEstimatorParamMaps(paramGrid)

# Get the best model
cvModel = crossVal.fit(trainSetDF).bestModel

predictionsAndLabelsDF = cvModel.transform(testSetDF).select("Atmospheric_Temperature", "Vacuum_Speed", "Atmospheric_Pressure", "Relative_Humidity", "Power_Output", "Predicted_PE")

# Create new rmse
rmseNew = regEval.evaluate(predictionsAndLabelsDF)

# Get R-Squared based on the new model
r2New = regEval.evaluate(predictionsAndLabelsDF, {regEval.metricName: "r2"})

# print("Regularization parameter of the best model: {0:.2f}".format(cvModel.stages[-1]._java_obj.parent().getRegParam()))


# Create a DecisionTree
decisionTree = DecisionTreeRegressor()
decisionTree.setLabelCol('Power_Output').setPredictionCol('Predicted_PE').setFeaturesCol('features').setMaxBins(100)
dtPipeline = Pipeline()

# Set stages
dtPipeline.setStages([vectorizer, decisionTree])
# print(decisionTree.maxDepth)

# Reuse the cross validator
crossVal.setEstimator(dtPipeline)

# Tune over dt.maxDepth parameter on the values 2 and 3, create a parameter grid using the ParamGridBuilder
paramGrid = ParamGridBuilder().addGrid(decisionTree.maxDepth, [2, 3]).build()

# Add grid to cross validator & get the best model
crossVal.setEstimatorParamMaps(paramGrid)
dtModel = crossVal.fit(trainSetDF).bestModel

predictionsAndLabelsDF = dtModel.transform(testSetDF).select("Atmospheric_Temperature", "Vacuum_Speed", "Atmospheric_Pressure", "Relative_Humidity", "Power_Output", "Predicted_PE")
rmseDT = regEval.evaluate(predictionsAndLabelsDF)
r2DT = regEval.evaluate(predictionsAndLabelsDF, {regEval.metricName: "r2"})

# print(rmseDT)
# print(r2DT)


# Create Random Forest
rf = RandomForestRegressor()
rf.setLabelCol("Power_Output").setPredictionCol("Predicted_PE").setFeaturesCol("features").setSeed(100088121).setMaxDepth(8).setNumTrees(30)
rfPipeline = Pipeline()

rfPipeline.setStages([vectorizer, rf])

crossVal.setEstimator(rfPipeline)

# Tune the rf.maxBins parameter on the values 50 and 100, create a parameter grid using the ParamGridBuilder
paramGrid = ParamGridBuilder().addGrid(rf.maxBins, [50, 100]).build()
crossVal.setEstimatorParamMaps(paramGrid)
rfModel = crossVal.fit(trainSetDF).bestModel

predictionsAndLabelsDF = rfModel.transform(testSetDF).select("Atmospheric_Temperature", "Vacuum_Speed", "Atmospheric_Pressure", "Relative_Humidity", "Power_Output", "Predicted_PE")
rmseRF = regEval.evaluate(predictionsAndLabelsDF)
r2RF = regEval.evaluate(predictionsAndLabelsDF, {regEval.metricName: "r2"})

print(rmseRF)
print(r2RF)