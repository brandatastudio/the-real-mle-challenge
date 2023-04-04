from flask import Flask ,jsonify, request, make_response
import sys
sys.path.append("/src/src")
import prediction_app.transform as pr_app_t
import yaml
import requests


app= Flask(__name__)


@app.route("/predictions", methods=["GET"])
def predict():

    data = request.get_json()
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
        prediction_text = pr_app_t.predict(config = config, data_dict = data)

    output = {'id': data['id'] , 'price_category': str(prediction_text)}
    return(make_response(jsonify(output)))


@app.route("/", methods=["GET"])
def hello():
    return jsonify("hello ,open a new shell for the python image,and execute `python prediction_app/test_api_call.py` to test inference")

    
if __name__=='__main__':
    app.run(host="0.0.0.0" , debug=True , port=3000)





