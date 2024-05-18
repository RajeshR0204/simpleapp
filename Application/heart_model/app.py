import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import gradio
from fastapi import FastAPI, Request, Response

import random
import numpy as np
import pandas as pd
import joblib as joblib


# FastAPI object
app = FastAPI()


file_path = "xgboost-model.pkl"
trained_model = joblib.load(filename=file_path)

# Function for prediction
from typing import Union
def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
  input_df=pd.DataFrame(input_data)
  results = trained_model.predict(input_df)
  return results

# data_in={'age':[75],'anaemia':[0],'creatinine_phosphokinase':[1786],'diabetes':[1],'ejection_fraction':[20],
           #     'high_blood_pressure':[0],'platelets':[14462000],'serum_creatinine':[1.1],'serum_sodium':[19990],'sex':[1],'smoking':[1],'time':[4]}

# UI - Input components
in_Pid = gradio.Textbox(lines=1, placeholder=None, value="22", label='age')
in_Pclass = gradio.Radio(['0', '1'], type="value", label='anaemia', value="0")
in_Pname = gradio.Textbox(lines=1, placeholder=None, value="1", label='creatinine_phosphokinase')
in_sex = gradio.Radio(["0", "1"], type="value", label='diabetes', value="0")
in_age = gradio.Textbox(lines=1, placeholder=None, value="1", label='ejection_fraction')
in_sibsp = gradio.Textbox(lines=1, placeholder=None, value="1", label='high_blood_pressure')
in_parch = gradio.Textbox(lines=1, placeholder=None, value="427000", label='platelets')
in_ticket = gradio.Textbox(lines=1, placeholder=None, value="1", label='serum_creatinine')
in_cabin = gradio.Textbox(lines=1, placeholder=None, value="138", label='serum_sodium')
in_embarked = gradio.Radio(["0", "1"], type="value", label='sex', value="0")
in_fare = gradio.Textbox(lines=1, placeholder=None, value="0", label='smoking')
in_time = gradio.Textbox(lines=1, placeholder=None, value="4", label='time')

# UI - Output component
out_label = gradio.Textbox(type="text", label='Prediction', elem_id="out_textbox")

# Label prediction function
def get_output_label(in_Pid, in_Pclass, in_Pname, in_sex, in_age, in_sibsp, in_parch, in_ticket, in_cabin, in_embarked, in_fare,in_time):

  data_in = {"age": [int(in_Pid)],
            "anaemia": [int(in_Pclass)],
            "creatinine_phosphokinase": [int(in_Pname)],
            "diabetes": [int(in_sex)],
            "ejection_fraction": [int(in_age)],
            "high_blood_pressure": [int(in_sibsp)],
            "platelets": [int(in_parch)],
            "serum_creatinine": [float(in_ticket)],
            "serum_sodium": [int(in_cabin)],
            "sex": [int(in_embarked)],
            "smoking": [int(in_fare)],
            "time": [int(in_time)]}

  #print (input_df)
  #result = make_prediction(input_data=input_df.replace({np.nan: None}))["predictions"]
  result = make_prediction(input_data=data_in)
  label = "Death Event" if result==1 else "No risk of Death Event"
  return label


# Create Gradio interface object
iface = gradio.Interface(fn = get_output_label,
                         inputs = [in_Pid, in_Pclass, in_Pname, in_sex, in_age, in_sibsp, in_parch, in_ticket, in_cabin, in_embarked, in_fare,in_time],
                         outputs = [out_label],
                         title="Heart Failure Survival Prediction",
                         description="Predictive model that answers the question: “What sort of people were more likely to survive heart failure?”",
                         allow_flagging='never'
                         )



# Mount gradio interface object on FastAPI app at endpoint = '/'
app = gradio.mount_gradio_app(app, iface, path="/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 
