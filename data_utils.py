import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split

## Functions used for data preprocessing and training in my Jupyter notebooks pertaining to logistic regression and KNN clustering for threat detection
## and data exploration.
def concat_dict(dict_to_concat, df):
    for key, val in dict_to_concat.items():
        df = pd.concat([df, dict_to_concat[key]],axis=1)
    return df

class data_processing:
    @staticmethod
    #function to perform data preprocessing - dropping unnecessary columns, one hot encoding
    def process_data(df, drop_cols, cat_cols):
        df2 = df.drop(drop_cols, axis=1)
        
        cats = df[cat_cols]
        dums = {col: pd.get_dummies(cats[col], drop_first=True) for col in cats.columns}
        df2 = df2.drop(cat_cols,axis=1)
        df3 = concat_dict(dums, df2)
        
        return df3

    @staticmethod
    def split_data(df, target_col):
        y = df[target_col]
        x = df.drop(target_col, axis=1)
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=11)
        return x_train, x_test, y_train, y_test