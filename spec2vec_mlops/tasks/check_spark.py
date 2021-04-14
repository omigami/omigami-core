from prefect import task
from pyspark.sql import SparkSession


@task
def check_spark(filepath: str) -> int:
    spark = SparkSession.builder.appName("testSparkAWS").getOrCreate()
    sc = spark.sparkContext

    hadoopConf = sc._jsc.hadoopConfiguration()
    hadoopConf.set("fs.s3a.multipart.size", "104857600")

    rdd = sc.textFile(filepath)
    return rdd.count()
