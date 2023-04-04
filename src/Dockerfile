FROM python:3.8

WORKDIR /src/src
ENV LISTEN_PORT=3000
EXPOSE 3000
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


#comment and uncomment the COPY and CMD to activate and deactivate prod mode
#COPY main.py ./
#CMD ["python", "main.py"]