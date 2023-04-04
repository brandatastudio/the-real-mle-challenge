import os

os.system("python data_prep/task.py")
os.system("python training/task.py")
os.system("python explainability/task.py")
os.system("python prediction_app/app.py")