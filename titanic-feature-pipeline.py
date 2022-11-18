import os
import modal
import numpy as np
from titaniccleaner import clean

LOCAL=False

if LOCAL == False:

   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4","joblib","seaborn","scikit-learn==0.24.2","dataframe-image"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("ScalableML_lab1"))
   def f():
       g()

def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()
    titanic_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")
    titanic_df = clean(titanic_df)
    print(titanic_df.head())
    titanic_fg = fs.get_or_create_feature_group(
        name="titanic_modal",
        version=1,
        primary_key=['class', 'sex', 'age', 'sibs', 'par_ch', 'fare', 'deck_1',
       'deck_2', 'deck_3', 'deck_4', 'deck_5', 'deck_6', 'deck_7',
       'embarked_1', 'embarked_2'], 
        description="Titanic dataset")
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
