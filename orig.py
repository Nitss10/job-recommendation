from time import time
from pyspark.sql import SparkSession
from pyspark.ml  import Pipeline
from pyspark.sql.functions import mean,col,split, regexp_extract, when, lit
from pyspark.sql.functions import col
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF

from pyspark.ml import Pipeline
from pyspark.ml.linalg import DenseVector, SparseVector
from pyspark.sql.functions import udf
from scipy.spatial.distance import cosine
from pyspark.sql.types import FloatType
from pyspark.sql.functions import lit
from pyspark.sql.functions import desc
import multiprocessing

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import matplotlib.pyplot as plt
#import seaborn as sns
from time import time
from pyspark.sql import SparkSession
from pyspark.ml  import Pipeline
from pyspark.sql import SQLContext
from pyspark.sql.functions import mean,split, regexp_extract, when, lit
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.feature import Word2Vec

spark = SparkSession.builder.appName('recommender_system').getOrCreate()

df_data=spark.read.csv( "marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv",
                          header=True,
                          inferSchema=True )

df_data.printSchema()

df_data = df_data.limit(4000)
print('Initial no of rows in data frame:',df_data.count())
print('Dropping rows with null values and empty strings')

# List of columns to check for null or empty values
columns = ['Job Title', 'Job Experience Required', 'Key Skills', 'Role Category', 'Location', 'Functional Area', 'Industry', 'Role']

# Drop rows with null values in any of the specified columns
df= df_data.na.drop(subset=columns)

# Further filter out rows with empty strings in any of the specified columns
for column in columns:
    df = df.filter(col(column) != '')

print('No of rows after dropping invalid data in data frame:',df.count())

print('No of cores:',multiprocessing.cpu_count())
df = spark.createDataFrame(df.rdd.repartition(4))

print('Repartitioning completed')


# # Replace null values with an empty string in each column
for col_name in columns:
    df = df.na.fill({col_name: ''})

## minDFs for CountVectorizer
minDFs = {'Job Title':1.0, 'Job Experience Required':1.0, 'Key Skills':1.0,'Role Category':1.0,'Location':1.0,'Functional Area':1.0,'Industry':1.0,'Role':1.0}
print('reached 3')

preProcStages = []

for colm in columns:
  regexTokenizer = RegexTokenizer(gaps=False, pattern='\w+', inputCol=colm, outputCol=colm+'Token')
  stopWordsRemover = StopWordsRemover(inputCol=colm+'Token', outputCol=colm+'SWRemoved')
  countVectorizer = CountVectorizer(minDF=minDFs[colm], inputCol=colm+'SWRemoved', outputCol=colm+'TF')
  idf = IDF(inputCol=colm+'TF', outputCol=colm+'IDF')
  word2Vec = Word2Vec(vectorSize=100, minCount=0, inputCol=colm+'SWRemoved', outputCol=colm+'W2V')
  preProcStages += [regexTokenizer, stopWordsRemover, countVectorizer, idf,word2Vec]


from pyspark.ml import Pipeline
pipeline = Pipeline(stages=preProcStages)
model = pipeline.fit(df)
df_comb= model.transform(df)

# Example user input
user_input = {
    "Job Title": "Software Engineer",
    "Job Experience Required": "3-5 years",
    "Key Skills": "Java, SQL",
    "Role Category": "Programming",
    "Location": "New York",
    "Functional Area": "IT Software",
    "Industry": "IT-Software",
    "Role": "Software Developer"
}

# Convert user input into a DataFrame
user_df = spark.createDataFrame([user_input])

# Process user input using the same pipeline model
transformed_user_df = model.transform(user_df)

# Function to convert to dense array
def to_dense_array(vector):
    if isinstance(vector, SparseVector):
        return DenseVector(vector.toArray()).toArray()
    elif isinstance(vector, DenseVector):
        return vector.toArray()
    return vector

#CHOOSE VECTOR TYPE HERE!!

vector_type="W2V" #IDF
# Get the user's vectors
user_job_title_vector = to_dense_array(transformed_user_df.first()["Job Title"+vector_type])
user_key_skills_vector = to_dense_array(transformed_user_df.first()["Key Skills"+vector_type])

# UDF to calculate cosine similarity
def cosine_similarity_with_arrays(vector1_array, vector2_array):
    vector1 = DenseVector(vector1_array)
    vector2 = DenseVector(vector2_array)
    return float(1 - cosine(vector1, vector2))

cosine_with_arrays_udf = udf(cosine_similarity_with_arrays, FloatType())

overall_start_time = time()

df_comb.show(5)


df_comb = df_comb.withColumn("job_title_similarity", cosine_with_arrays_udf(lit(user_job_title_vector), col("Job Title"+vector_type))) \
                 .withColumn("key_skills_similarity", cosine_with_arrays_udf(lit(user_key_skills_vector), col("Key Skills"+vector_type)))

start_time = time()
print('Starting cosine calc:',start_time)
df.count()
end_time = time()
print('Ending cosine calc:',end_time)
print('Time taken to cosine calc:',end_time-start_time)

threshold=0.4
df_comb=df_comb.filter((col("job_title_similarity") > threshold) | (col("key_skills_similarity") > threshold))
df_comb = df_comb.withColumn("total_similarity", (col("job_title_similarity") + col("key_skills_similarity")) / 2)

top_10_jobs = df_comb.orderBy("total_similarity", ascending=False).limit(10)

start_time = time()
print('Starting top 10 select calc:',start_time)
top_10_jobs.select("Uniq Id", "Job Title", "Location", "total_similarity").show()
end_time = time()
print('Ending top 10 select calc:',end_time)
print('Time taken to top 10 select calc:',end_time-start_time)

overall_end_time = time()
print("Overall time taken:",overall_end_time-overall_start_time)
