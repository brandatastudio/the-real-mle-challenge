FROM python:3.8

WORKDIR /src/src
ENV LISTEN_PORT=3000
EXPOSE 3000
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
#ENTRYPOINT [ "main.py" ] for when we want the image to run as productive