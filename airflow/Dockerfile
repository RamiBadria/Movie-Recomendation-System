FROM apache/airflow:2.5.0

ARG DOCKER_GID=999
USER root
RUN apt-get update && \
    apt-get install -y default-libmysqlclient-dev && \
    apt-get clean && \
    groupadd -g ${DOCKER_GID} docker && \
    usermod -aG docker airflow
USER airflow
COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt