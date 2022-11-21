"""
Cleaner for the titanic dataset
Based on https://ryanwingate.com/projects/machine-learning-data-prep/titanic/titanic-cleaning/
Ziyou Li
"""

import pandas as pd
import numpy as np


def clean(titanic):
    ## Read Data
    t = titanic.copy()
    # t.head()


    ## Clean Data

    # Cabin is only useful to obtain the Deck.
    t['Deck'] = t['Cabin'].str.extract(r'([A-Z])?(\d)')[0]

    # Ticket, Name, PassengerID are all not useful.
    t.reset_index(drop=True)
    t['idx'] = t.index
    t = t[['idx','Pclass','Sex','Age','SibSp','Parch','Fare','Deck','Embarked','Survived']].copy()
    t.columns = ['idx','pclass','sex','age','sibs','par_ch','fare','deck','embarked','survived']

    # Deal with NaN
    # Fill NaN values except for deck and embarked with the mean
    t['age'] = t['age'].fillna(t['age'].mean())
    t['fare'] = t['fare'].fillna(t['fare'].mean())

    # Fill NaN values of embarked with the most common value, S
    t['embarked'] = t['embarked'].fillna('S')

    # Fill Deck with 0
    t['deck'] = t['deck'].fillna(0)
    # t.head()


    ## Convert Categorical (nominal, not ordinal) data to Numeric data

    t['pclass raw'] = t['pclass']
    t['sex raw'] = t['sex']
    t['par_ch raw'] = t['par_ch']
    t['fare raw'] = t['fare']
    t['deck raw'] = t['deck']
    t['embarked raw'] = t['embarked']
    t['survived raw'] = t['survived']

    t['pclass'] = t['pclass'].astype('category')
    t['pclass'] = t['pclass'].cat.codes
    t[['pclass','pclass raw','idx']].groupby(['pclass',
                                            'pclass raw']).count()
    
    t['sex'] = t['sex'].astype('category')
    t['sex'] = t['sex'].cat.codes
    t[['sex','sex raw','idx']].groupby(['sex',
                                        'sex raw']).count()
                    
    t['deck'] = t['deck'].astype('category')
    t['deck'] = t['deck'].cat.codes
    t[['deck','deck raw','idx']].groupby(['deck',
                                      'deck raw']).count()

    t['embarked'] = t['embarked'].astype('category')
    t['embarked'] = t['embarked'].cat.codes
    t[['embarked','embarked raw','idx']].groupby(['embarked',
                                              'embarked raw']).count()

    t = t[['idx','pclass','sex','age','sibs','par_ch','fare','deck','embarked','survived']].copy()
    # print(t.shape)
    # t.head()

    ## One-Hot (Dummy) Encode Categoricals (nominal, not ordinal)

    t = pd.get_dummies(t,
                   columns=['deck',
                            'embarked'],
                   drop_first=True)
    t = t[['pclass', 'sex', 'age', 
        'sibs', 'par_ch', 'fare',
        'deck_1', 'deck_2', 'deck_3', 'deck_4', 'deck_5', 'deck_6', 'deck_7',
        'embarked_1', 'embarked_2', 
        'survived']].copy()

    return t
    
