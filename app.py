import urllib
import json
import os
import  pandas as pd

from flask import Flask
from flask import request
from flask import make_response
import pickle

# Flask app should start in global layout
app = Flask(__name__)
model=pickle.load(open('model.pkl','rb')) 
 
@app.route('/webhook', methods=['POST'])
def webhook():
        req = request.get_json(silent=True, force=True)
        res = json.dumps(req['queryResult']['parameters'], indent=4)
        int_features=json.loads(res)
        print(int_features)
        final_features=pd.DataFrame({'age':[int(int_features['age']['amount'])],'sex':[int(int_features['sex'])],'cp':[int(int_features['cp'])],'trestbps':[int(int_features['trestbps'])],'chol':[int(int_features['chol'])],'fbs':[int(int_features['fbs'])],'restecg':[int(int_features['restecg'])],'thalach':[int(int_features['thalach'])],'exang':[int(int_features['exang'])],'oldpeak':[float(int_features['oldpeak'])],'slope':[float(int_features['slope'])],'ca':[int(int_features['ca'])],'thal':[int(int_features['thal'])]})
        prediction=model.predict(final_features)
        pred=((prediction[0][1]*0.842)*100)
        r=get_data(prediction)
        print(r)
        r=json.dumps(r)
        result = make_response(r)
        result.headers['Content-Type'] = 'application/json'
        return result
     
 
def get_data(prediction):
   output=prediction[0][1]*100
   if output >=50:
        pred='You have '+ str(output)+'%'+' possibility of having heart disease.Please take immediate action to cure your self.'
   elif 30<=output<=50:
        pred='You have '+ str(output)+'%'+' possibility of having heart disease.Please take any actions and precautions to not to get a heart disease.'
   elif 0<output<=30:
        pred='You have '+ str(output)+'%'+' possibility of having heart disease.Anyway you have to be careful.'
   elif 0==output:
        pred='You have '+ str(output)+'%'+' possibility of having heart disease.You are safe'
 
    
   return {
       "fulfillmentText" : pred,
        
   }
 
if __name__ == '__main__':
    port = int(os.getenv('PORT', 80))

    print ("Starting app on port %d" %(port))

    app.run(debug=True, port=port, host='0.0.0.0')
