FROM python:3.8

RUN pip install mlflow==2.0.1
RUN pip install psycopg2

EXPOSE 5000

RUN mkdir -p /server
WORKDIR /server
COPY . /server

RUN chmod +x wait-for-it.sh

RUN mkdir /mlflow/

CMD ./wait-for-it.sh postgres:5432 -- mlflow server \
    --backend-store-uri postgresql://admin:secret@postgres:5432/mlflow \
    --default-artifact-root /mlflow \
    --host 0.0.0.0