import pandas as pd
import titaniccleaner as tc

titanic_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")
titanic_df = tc.clean(titanic_df)