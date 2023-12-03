import streamlit as st
from time import time
from pyspark.sql import SparkSession
from pyspark.ml  import Pipeline
from pyspark.sql.functions import lit
from pyspark.sql.functions import col
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml import Pipeline
from pyspark.ml.linalg import DenseVector, SparseVector
from pyspark.sql.functions import udf
from scipy.spatial.distance import cosine
from pyspark.sql.types import FloatType
import multiprocessing
from pyspark.ml.feature import Word2Vec
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('recommender_system').getOrCreate()

# Function to convert to dense array
def to_dense_array(vector):
    if isinstance(vector, SparseVector):
        return DenseVector(vector.toArray()).toArray()
    elif isinstance(vector, DenseVector):
        return vector.toArray()
    return vector

# UDF to calculate cosine similarity
def cosine_similarity_with_arrays(vector1_array, vector2_array):
    vector1 = DenseVector(vector1_array)
    vector2 = DenseVector(vector2_array)
    return float(1 - cosine(vector1, vector2))

# Function to get recommendations
#CHOOSE VECTOR TYPE HERE!!
#vector_type="W2V" # or IDF
def get_recommendations(user_input,vector_type='W2V'):
    df_data=spark.read.csv( "marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv",
                          header=True,
                          inferSchema=True )
    df_data.printSchema()

    #df_data = df_data.limit(1000)
    print('Initial no of rows in data frame:',df_data.count())
    print('Dropping rows with null values and empty strings')

    # List of columns to check for null or empty values
    columns = ['Job Title', 'Job Experience Required', 'Key Skills', 'Role Category', 'Location', 'Functional Area', 'Industry', 'Role']

    # Drop rows with null values in any of the specified columns
    df= df_data.na.drop(subset=columns)
    # Further filter out rows with empty strings in any of the specified columns
    for column in columns:
        df = df.filter(col(column) != '')
    df = df.dropDuplicates()
    
    print('No of rows after dropping invalid data/duplicate in data frame:',df.count())

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


    pipeline = Pipeline(stages=preProcStages)
    model = pipeline.fit(df)
    df_comb= model.transform(df)

    # Convert user input into a DataFrame
    user_df = spark.createDataFrame([user_input])
    # Process user input using the same pipeline model
    transformed_user_df = model.transform(user_df)
    # Get the user's vectors
    user_job_title_vector = to_dense_array(transformed_user_df.first()["Job Title"+vector_type])
    user_key_skills_vector = to_dense_array(transformed_user_df.first()["Key Skills"+vector_type])
    user_location_vector = to_dense_array(transformed_user_df.first()["Location"+vector_type])
    cosine_with_arrays_udf = udf(cosine_similarity_with_arrays, FloatType())

    df_comb = df_comb.withColumn("job_title_similarity", cosine_with_arrays_udf(lit(user_job_title_vector), col("Job Title"+vector_type))) \
                 .withColumn("location_similarity", cosine_with_arrays_udf(lit(user_location_vector), col("Location"+vector_type))) \
                 .withColumn("key_skills_similarity", cosine_with_arrays_udf(lit(user_key_skills_vector), col("Key Skills"+vector_type)))

    threshold=0.4
    df_comb=df_comb.filter((col("job_title_similarity") > threshold) | (col("key_skills_similarity") > threshold) | (col("location_similarity") > threshold))
    df_comb = df_comb.withColumn("total_similarity", (col("job_title_similarity") + col("key_skills_similarity") + col("location_similarity")) / 3)


    top_10_jobs = df_comb.orderBy("total_similarity", ascending=False).limit(10)

    return top_10_jobs
  


# Streamlit UI
def main():
    st.title("Job Recommendation System")

    # User input fields
    job_title = st.text_input("Job Title")
    experience = st.text_input("Experience")
    key_skills = st.text_input("Key Skills")
    role_category = st.text_input("Role Category")
    location = st.text_input("Location")
    functional_area = st.text_input("Functional Area")
    industry = st.text_input("Industry")
    role = st.text_input("Role")

    if st.button("Recommend Jobs"):
        user_input = {
            "Job Title": job_title,
            "Job Experience Required": experience,
            "Key Skills": key_skills,
            "Role Category": role_category,
            "Location": location,
            "Functional Area": functional_area,
            "Industry": industry,
            "Role": role
        }

        recommendations = get_recommendations(user_input)
        final=recommendations.select("Uniq Id", "Job Title", "Location", "Key Skills", "total_similarity")
        final.show()
        st.dataframe(final, use_container_width=True)
        # st.write(final)

if __name__ == "__main__":
    main()