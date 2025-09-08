import pandas as pd 
import numpy as np 
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import chi2_contingency

## Functions used for data preprocessing and training in my Jupyter notebooks pertaining to logistic regression and KNN clustering for threat detection
## and data exploration.

# A helper function for data processing - concatenates the dataframes in a dictionary together.
def concat_dict(dict_to_concat, df):
    for key, val in dict_to_concat.items():
        df = pd.concat([df, dict_to_concat[key]],axis=1)
    return df

class data_processing:
    @staticmethod
    #Function to perform data preprocessing - dropping unnecessary columns and performing one hot encoding
    def process_data(df, drop_cols, cat_cols):
        df2 = df.drop(drop_cols, axis=1)
        
        cats = df[cat_cols] #get categorical columns
        dums = {col: pd.get_dummies(cats[col], drop_first=True) for col in cats.columns} #one hot encoding
        df2 = df2.drop(cat_cols,axis=1) #drop category columns with text
        df3 = concat_dict(dums, df2) #concatenate
        
        return df3

    @staticmethod
    #Function that trains based on features which have a statistically siginificant relationship with the dependent variable
    def select_feats_chi2(x_train, y_train):
        chi_l = list() #initiate list of cols with a statistically significant relationship
        for col in x_train.columns:
            contingency_table = pd.crosstab(x_train[col],y_train)
            chi2_statistic, p_value, degrees_of_freedom, expected_frequencies = chi2_contingency(contingency_table)
            if p_value <= 0.05:
                chi_l.append(col)
                
        clf_chi = LogisticRegression(random_state=0).fit(x_train[chi_l],y_train) #train logistic regression model
        return chi_l, clf_chi #return columns used and model
    
    @staticmethod
    #Function for splitting data into train, test, and x/y datasets
    def split_data(df, target_col):
        y = df[target_col] #define target col dataset
        x = df.drop(target_col, axis=1) #drop target col from independent dataset
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=11,stratify=y)
        return x_train, x_test, y_train, y_test

    @staticmethod
    #Function that checks whether or not the values of two groups are equal, if not, performs undersampling of the majority class.
    #Note that this only works for two groups, currently.
    def check_group_distr(df, group_col, count_col):
        temp = df.groupby(group_col)[count_col].count() #temporary dataframe grouped by the specified column
        if temp[0] != temp[1]:
            print('Labeled categories uneven, undersampling majority category...')
            rus = RandomUnderSampler(random_state=42)
            x,y = rus.fit_resample(df, df[group_col]) #undersample
            
            return x
        else:
            return df

    @staticmethod    
    #Function that checks whether or not the values of two groups are equal, if not, performs oversampling of the minority class.
    #Note that this only works for two groups, currently.
    def check_group_distr_over(df, group_col, count_col):
        temp = df.groupby(group_col)[count_col].count() #temporary dataframe grouped by the specified column
        if temp[0] != temp[1]:
            print('Labeled categories uneven, oversampling minority category...')
            smote = SMOTE(random_state=42)
            # Apply SMOTE to resample the dataset
            x,y= smote.fit_resample(df, df[group_col]) #oversample
            
            return x
        else:
            return df