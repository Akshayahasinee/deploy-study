import pandas as pd
from flask import Flask, request, jsonify
from waitress import serve
import pickle
# install flask, waitress, pickle into your ananconda environment

app = Flask(__name__)
# load logistics regression model to the server, 'rb' - read binary
logReg = pickle.load(open('logRegModel.pkl', 'rb'))
df_time_series = pd.read_pickle('moving_average2.pkl') 

@app.route('/logreg', methods=['GET'])
def callLogRegModel():
    xValue = request.args.get('x', type= float)
    yValue = request.args.get('y', type= float)
    return str(logReg.predict([[xValue, yValue]])[0])

@app.route('/timeseries', methods=['GET'])
def callModelFour():
   xValue = request.args.get('x', type= int)
   print(df_time_series[xValue])
   return str(df_time_series[xValue])

# run the server
if __name__ == '__main__':
    print("Starting the server.....")
    serve(app, host="0.0.0.0", port=8080)
