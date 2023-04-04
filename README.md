
# About this assignment:

-Task1 can be observed in the project structure and the code,
-Task2 is stored in the directory prediction_app.
-Task3 can be observed through the docker files and docker-compose

## Motivation behind this package: Intended use

This package was thought as a data science environment that serves both for experimentation with proper ml-ops and data-ops tracking, making use 
of locally stored postgres databases for mlflow experiment tracking, and dvc for data version tracking. Code is properly decoupled in tasks,(maybe to be properly scheduled with airflow in a later iteration). Each task is a subdirectory, formed by two files: 
 a task.py and transform.py, for CI/CD , a test.py file can be added to the task folders. An example of this can be found in prediction_app

Prediction_app being an app for deployment and basically, the representation of the exploitation layer, has it's main executing file called app.py to work better with flask, it's the only exception that doesn't call it's main executor "task.py". 


## Project structure:
three main folders, each representing a service in docker-compose file
-mlflow_docker:mlflow configuration to make sure it connects with postgres
-postgres_docker:data base that serves as a remote server hosted locally to be able to use mlflow ui through docker
-src: the main python image that will execute our training, the most important folder 

-postgres-store: Automatically generated when we launch postgres-docker image, serves as volume to connect local storage with the database/mlflow storage. Thanks to it, we can persist different container ml experiments and compare them when launching mlflow
#### src folder structure
    -data_prep:first  task in pipeline
    -training:second task in pipeline
    -explainability:third task in pipeline
    -prediction_app: last task, exploitation layer, app for inference

    -data: data files are stored here
    -models: productive winner model from the gridsearch experiments, and it's details are stored here
#### Files in src
    -utility.py: a  helping function used mainly in inference, although other functions (for example for cloud connection) could be stored here
    -main.py: pipeline execution for automation, used in productive mode (production mode can be activating by removing comment from src dockerfile on ENTRYPOINT )
    -dvc.yaml: pipeline configuration to execute through dvc, used in experimentation mode
    -config.yaml:training configuration file
    -dvc.lock:automatically generated dvc file, when we run dvc repro
    -utility.py: a  helping function used mainly in inference, although other functions (for example for cloud connection) could be stored here

### Structure of task.py

 Task.py are constructed mainly basing each stage as an etl process, mainly three classes are created, read, task and load, this is so to facilitate decoupling of the code in an exectuded scheduled dag that is dependent of a cloud environment. Each task would read the data that depends on , transform it using mainly functions from transform.py and then load it to the local server folder or the cloud. 
 the idea is that, functions only used by that task are stored in transform. Even so, the code is organized so that different folders can use different functions of parallel folders. 

## Starting up the project:

If we just want to execute the current algorithm 
After cloning the repo, you can just run:

`docker-compose build`

`docker-compose up`

This will create and start running the docker images, after that, we can access the main application, the ds_environment service, with a command like this 

-Getting inside the ds environment: 

`docker exec -it the-real-mle-challenge-ds_environment-1 bash`

From that point on, we are inside our ds_environment we can use to train and store experiments. To modify and test changes, we can just edit code in src through our local text editor, and through the volumes that have been configured in the dockerfiles, the docker container image will edit src instantaneusly to represent our local changes. If we modify the docker image or the docker compose, we need to relaunch docker-compose build for the specific container affected. 

from here, we can run a training experiment, we just initiate dvc from root folder

`dvc init`

and run from src/

`dvc repro`

IMPORTANT: make sure to specify an experiment run in the config file, before calling dvc repro for second time, otherwise it will cause an error because the same experiment run name can't be used. If you are debugging, specify in config file the experiment_name parameter as False

## How dvc works: 
to use dvc basically one thing is necesary, doing dvc init from root folder. After that dvc repro will take care of executing the python scripts
that represent the data pipeline (defined in dvc.yaml), and storing the information in dvc.lock file, this file is tracked as an artifact in mlflow along with the git id version

-dvc repro from src folder, will execute the dvc pipeline defined in dvc.yaml with all the steps of the code. 

This dvc.yaml serves as great documentation of dependencies between the projects, and how it's executed (basically when te project is automatized, it executes 4 tasks in sequential order data_prep --> training ---> explainability --> prediciton_app as sequential stages, each directory is structured with a task.py file that executes the task and a transform.py file )

Although organized for automatization, the same project can serve as a ml experimentation environment, making full use of local postgres container to store mlflow runs, a config file where training configurations can be placed to execute in gridsearch, and dvc to properly register data versions. 


## Launching mlflow
after activating image, you can just execute `mlflow ui` and you will see the mlflow ui interface with experiment informaiton, 
you can also launch the image from the web through visual studio code extension, the ui will be accessible to port host 5000
## Debugging tips:

`docker-compose build {servicename}` can be used if we modify only one dockerfile, to rebuild only the affected service, the best example and most common use case would be `docker-compose build {servicename}` to rebuild image after modifying  ds_environment dockerfile (the src folder), but we can use this command with any of the services specified in the docker-compose.yml


## Productivizing the image:

As cloned, the image is mainly prepared as a data_science experimentation environment, this is so because that's mainly what the code inherited from the task assignment was used for, and I found interesting to focus the project this way, to simulate deployment we just need to remove the comments in src dockerfile, "CMD", this will execute data preparation, training, and flask app deployment in a go by the execution of main.py




Notes:
-docker-compose build creates a local directory in root branch called postgres_store, that serves as local remote database for mlflow
to delete it it's necesary to shutdown the imagebefore hand, if not, the system will consider the files as being opened and edited and prohibit their deletion

-When working as a laboratory, it's preferable to execute training runs through `dvc repro`because this way all model dand data is stored as part of the experiment in mlflow, allowing for accurate reproducibility in the future. 


## Code format:

The black python library was used for formatting