from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
import logging
import os
import sparknlp

def create_spark_session(app_name="CPU_Embeddings"):
    # Konfiguracja Spark z wsparciem dla SparkNLP
    spark = SparkSession.builder \
        .appName("Spark NLP") \
        .master("local[6]") \
        .config("spark.driver.memory", "14G") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "2000M") \
        .config("spark.driver.maxResultSize", "0") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.1") \
        .config("spark.memory.fraction", 0.8) \
        .config("spark.memory.storageFraction", 0.3) \
        .config("spark.sql.shuffle.partitions", 12) \
        .config("spark.default.parallelism", 12) \
        .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:+UseCompressedOops") \
        .config("spark.driver.cores", "6") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.driver.extraClassPath", "local_jars/*") \
        .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true") \
        .config("spark.worker.cleanup.enabled", "true") \
        .getOrCreate()
    

    return spark

def create_optimized_spark_session(app_name="Memory_Optimized_App"):
    return SparkSession.builder \
        .appName(app_name) \
        .master("local[4]") \
        .config("spark.driver.memory", "10G") \
        .config("spark.driver.maxResultSize", "2G") \
        .config("spark.memory.fraction", 0.7) \
        .config("spark.memory.storageFraction", 0.4) \
        .config("spark.sql.shuffle.partitions", 8) \
        .config("spark.default.parallelism", 8) \
        .config("spark.driver.cores", "4") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.cleaner.periodicGC.interval", "1min") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "2g") \
        .config("spark.sql.autoBroadcastJoinThreshold", "10m") \
        .config("spark.sql.files.maxPartitionBytes", "128m") \
        .config("spark.driver.extraJavaOptions", 
                "-XX:+UseG1GC -XX:+UseCompressedOops -XX:+PrintGCDetails -XX:G1HeapRegionSize=16M") \
        .getOrCreate()
    

    return spark


def load_csv_to_df(spark, file_path, schema=None, header=True, delimiter=",", inferSchema=True, quote='"', escape = '"'):
    """
    Loads a CSV file into a PySpark DataFrame with specified options
    
    Parameters:
    -----------
    spark : SparkSession
        Active Spark session
    file_path : str
        Path to the CSV file
    schema : StructType, optional
        Custom schema for the DataFrame
    header : bool, default=True
        Whether the CSV has a header row
    delimiter : str, default=","
        CSV delimiter character
    inferSchema : bool, default=True
        Whether to automatically infer data types
        
    Returns:
    --------
    pyspark.sql.DataFrame
        Loaded DataFrame
    """
    try:
        # Basic read options
        read_options = {
            "quote" : quote,
            "escape" : escape,
            "header": str(header).lower(),
            "delimiter": delimiter,
            "inferSchema": str(inferSchema).lower()
        }
        
        # Create DataFrame reader
        reader = spark.read.format("csv").options(**read_options)
        
        # Apply schema if provided
        if schema:
            reader = reader.schema(schema)
            
        # Load the DataFrame
        df = reader.load(file_path)
        
        logging.info(f"Successfully loaded CSV from {file_path}")
        print(f"DataFrame loaded with {df.count()} rows and {len(df.columns)} columns")
        
        return df
    
    except Exception as e:
        logging.error(f"Error loading CSV: {str(e)}")
        raise


def get_dataframe_dimensions(df):
    """
    Returns the dimensions (number of rows, number of columns) of a PySpark DataFrame
    
    Parameters:
    -----------
    df : pyspark.sql.DataFrame
        Input DataFrame
        
    Returns:
    --------
    tuple : (num_rows, num_columns)
        Tuple containing number of rows and columns
    """
    num_rows = df.count()
    num_cols = len(df.columns)
    return (num_rows, num_cols)
    
def print_dataframe_info(df):
    """
    Prints detailed information about DataFrame dimensions and schema
    
    Parameters:
    -----------
    df : pyspark.sql.DataFrame
        Input DataFrame
    """
    num_rows, num_cols = get_dataframe_dimensions(df)
    print(f"Number of rows: {num_rows}")
    print(f"Number of columns: {num_cols}")
    print("\nColumn names:")
    for col in df.columns:
        print(f"- {col}")
    print("\nSchema:")
    df.printSchema()
