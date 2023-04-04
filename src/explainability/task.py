
import sys
sys.path.append("/src/src")
import explainability.transform as exp_t
import pickle
import yaml
import pandas as pd
from time import gmtime, strftime




class read():

    def __init__(self, 
                config:dict):
                self.config = config

    def get_model_details(self)  -> dict:
        
        winning_model_details =  pickle.load(open(config['training']['ml_model_details_path'], 'rb'))

        return(winning_model_details)

    
    def get_data(self, sep :str = ','):

        '''reads csv raw data and generates pandas df'''
        
        processed_data = pd.read_csv(filepath_or_buffer = self.config['training']['prepped_for_Ml_data_path'], sep = sep)
        raw_data = pd.read_csv(filepath_or_buffer = self.config['data_prep']['raw_data_path'], sep = sep)
        return(processed_data , raw_data)





















if __name__ == "__main__":


    with open('config.yaml') as f:
     
    
        try:
            config = yaml.safe_load(f)

            read = read(config = config)

            winning_model_details = read.get_model_details()
            processed_data , raw_data = read.get_data()
            sys.stdout.write("explainability read log: Data and winning model properly read" + "|Time of run:" +
                  str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())))

        except Exception as e:
            sys.stderr.write("explainability read log: data and winning model read failed |error message:"+ str(e) + "|Time of run:" +
                     str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())))
            
        try:
            exp_t.plot_feature_importance(winning_model_details)
            exp_t.plot_confusion_matrix(winning_model_details , config)
            exp_t.metric_evaluation_plots(winning_model_details , config)
            exp_t.plot_histogram(data = processed_data, winning_model_details = winning_model_details, column_to_plot = 'price')
            exp_t.plot_neighbourhood(data = processed_data)
            
            sys.stdout.write("explainability plotting log: plots made correctly" + "|Time of run:" +
                  str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())))
            
        except Exception as e:
            sys.stderr.write("explainability plottinh log: plotting failed |error message:"+ str(e) + "|Time of run:" +
                     str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())))
            
            


