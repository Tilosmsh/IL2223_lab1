import gradio as gr
import numpy as np
from PIL import Image
import requests

import hopsworks
import joblib

project = hopsworks.login(api_key_value="CDqcnm3gyfxjyCO8.TZwOClLOwCqDp33vX0P5Q2nsvNNyEhfBMArwNoPjnb9tUSSKq6I8X35HQ5D2tlJ7")
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("titanic_modal", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_modal.pkl")


def titanic(pclass, sex, age, sibs, par_ch, fare):
    input_list = []
    input_list.append(pclass)
    input_list.append(sex)
    input_list.append(age)
    input_list.append(sibs)
    input_list.append(par_ch)
    input_list.append(fare)
    input_list.extend(list(np.random.choice([0,1], 9)))
    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(input_list).reshape(1, -1)) 
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
    man_url = "https://raw.githubusercontent.com/Tilosmsh/IL2223_lab1/main/images/" + ("survived.jpg" if res[0]==1 else "dead.jpg")
    img = Image.open(requests.get(man_url, stream=True).raw)            
    return img
        
demo = gr.Interface(
    fn=titanic,
    title="Titanic Predictive Analytics",
    description="Experiment with passenger class, sex, age, number of siblings, number of parents & children and fare, to predict whether the passenger survived.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=1, label="Passenger Class (0, 1 or 2)"),
        gr.inputs.Number(default=1, label="Sex (0 or 1)"),
        gr.inputs.Number(default=30.0, label="Age (0 to 80)"),
        gr.inputs.Number(default=1, label="Number of Siblings (0 to 8)"),
        gr.inputs.Number(default=1, label="Number of Parents and children (0 to 6)"),
        gr.inputs.Number(default=35.0, label="Fare (0 to 513)"),
        ],
    outputs=gr.Image(type="pil"))

demo.launch()
