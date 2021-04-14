FROM continuumio/miniconda3
ENV PATH="/opt/conda/bin/:${PATH}"
WORKDIR /opt/spec2vec_mlops

COPY . /opt/spec2vec_mlops
RUN apt-get -y install curl
RUN head -n -3 requirements/environment.frozen.yaml | sed 's/spec2vec_mlops/base/g' > environment-docker.yml

RUN /opt/conda/bin/conda env update --file environment-docker.yml

RUN pip install -e /opt/spec2vec_mlops

# HADOOP

ENV HADOOP_VERSION 2.7.2
ENV HADOOP_HOME /usr/hadoop-$HADOOP_VERSION
ENV HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
ENV PATH $PATH:$HADOOP_HOME/bin
RUN curl -sL --retry 3 \
    "http://archive.apache.org/dist/hadoop/common/hadoop-$HADOOP_VERSION/hadoop-$HADOOP_VERSION.tar.gz" \
    | gunzip \
    | tar -x -C /usr/ && \
    rm -rf $HADOOP_HOME/share/doc

# SPARK

ENV SPARK_VERSION 2.0.0
ENV SPARK_PACKAGE spark-$SPARK_VERSION-bin-without-hadoop
ENV SPARK_HOME /usr/spark-$SPARK_VERSION
ENV PYSPARK_PYTHON python3
ENV PYSPARK_DRIVER_PYTHON ipython
ENV SPARK_DIST_CLASSPATH="$HADOOP_HOME/etc/hadoop/*:$HADOOP_HOME/share/hadoop/common/lib/*:$HADOOP_HOME/share/hadoop/common/*:$HADOOP_HOME/share/hadoop/hdfs/*:$HADOOP_HOME/share/hadoop/hdfs/lib/*:$HADOOP_HOME/share/hadoop/hdfs/*:$HADOOP_HOME/share/hadoop/yarn/lib/*:$HADOOP_HOME/share/hadoop/yarn/*:$HADOOP_HOME/share/hadoop/mapreduce/lib/*:$HADOOP_HOME/share/hadoop/mapreduce/*:$HADOOP_HOME/share/hadoop/tools/lib/*"
ENV PATH $PATH:$SPARK_HOME/bin
RUN curl -sL --retry 3 \
    "http://d3kbcqa49mib13.cloudfront.net/$SPARK_PACKAGE.tgz" \
    | gunzip \
    | tar x -C /usr/ && \
    mv /usr/$SPARK_PACKAGE $SPARK_HOME && \
    rm -rf $SPARK_HOME/examples $SPARK_HOME/ec2

WORKDIR /$SPARK_HOME
CMD ["bin/spark-class", "org.apache.spark.deploy.master.Master"]