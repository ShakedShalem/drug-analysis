import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data/druguse.tsv', sep='\t')


def create_features_df(df):
    X = df[['IRALCFY', 'IRMJFY', 'IRCOCFY', 'IRCRKFY', 'IRHERFY', 'IRHALFY', 'IRINHFY', 'IRANLFY',
            'IRTRNFY', 'IRSTMFY', 'IRSEDFY', 'IRCIGFM',
            'CIGYR', 'CGRYR', 'SNFYR', 'ALCYR', 'CRKYR', 'SUMYR',
            'IRCIGAGE', 'IRCGRAGE', 'IRSNFAGE', 'IRALCAGE', 'IRCRKAGE', 'SUMAGE',
            'DEPNDMRJ', 'DPILLALC', 'ABODILAL', 'TOBFLAG', 'SNFFLAG', 'ALCFLAG', 'CRKFLAG', 'SUMFLAG',
            'EDUCCAT2', 'CATAG6', 'IRSEX', 'INCOME', 'NEWRACE2']]
    return X


def create_target(df):
    y = df[['AMDEYR']]
    # Change labels to 1 and 0
    mask = y[y.AMDEYR == 2]
    rows_to_change = list(mask.index.values)
    y.set_value(rows_to_change, 'AMDEYR', 0)
    return y


def clean_data(X, y):
    # Create a mask to remove rows 'Aged 12-17'
    mask = y[y.AMDEYR < 0]
    rows_to_remove = list(mask.index.values)
    X = X.drop(X.index[rows_to_remove])

    # Reshape y
    y = y[y.AMDEYR > -1]
    y = y.T.squeeze()
    return X, y


#X = create_features_df(df)
#y = create_target(df)
#X, y = clean_data(X, y)

new_X = df[['IRALCFY', 'IRMJFY', 'IRCOCFY', 'IRCRKFY', 'IRHERFY', 'IRHALFY', 'IRINHFY', 'IRANLFY',
            'IRTRNFY', 'IRSTMFY', 'IRSEDFY', 'IRCIGFM',
            'CIGYR', 'CGRYR', 'SNFYR', 'ALCYR', 'CRKYR', 'SUMYR',
            'IRCIGAGE', 'IRCGRAGE', 'IRSNFAGE', 'IRALCAGE', 'IRCRKAGE', 'SUMAGE',
            'DEPNDMRJ', 'DPILLALC', 'ABODILAL', 'TOBFLAG', 'SNFFLAG', 'ALCFLAG', 'CRKFLAG', 'SUMFLAG',
            'EDUCCAT2', 'CATAG6', 'IRSEX', 'INCOME', 'NEWRACE2']]

# reduce_X = df[['IRALCFY', 'IRALCAGE', 'IRCIGAGE', 'SUMAGE', 'IRCGRAGE', 'EDUCCAT2',
#               'IRMJFY', 'INCOME', 'IRCIGFM', 'CATAG6', 'IRANLFY', 'IRTRNFY',
#               'IRSNFAGE', 'IRHALFY', 'IRSTMFY', 'DPILLALC', 'IRCOCFY', 'IRSEX',
#               'IRINHFY', 'ABODILAL', 'CIGYR', 'IRCRKAGE', 'CGRYR', 'IRSEDFY']]

# Change values of 991 and 993 on frequency of use
freq_cols = ['IRALCFY', 'IRMJFY', 'IRCOCFY', 'IRCRKFY', 'IRHERFY', 'IRHALFY', 'IRINHFY', 'IRANLFY',
             'IRTRNFY', 'IRSTMFY', 'IRSEDFY', 'IRCIGFM']

# freq_cols = ['IRALCFY', 'IRMJFY', 'IRCOCFY', 'IRHALFY', 'IRINHFY', 'IRANLFY',
#             'IRTRNFY', 'IRSTMFY', 'IRSEDFY', 'IRCIGFM']

# for col in freq_cols:
#    X[col].replace(to_replace=[991, 993], value=[-2, -1], inplace=True)

# Dummies the Race feature
#race_columns = pd.get_dummies(X.NEWRACE2).iloc[:, 1:]

#X[race_columns.columns] = race_columns

#reduce_X, y = clean_data(reduce_X, y)

# X.to_pickle('feature_df.pkl')
# y.to_pickle('target.pkl')
# reduce_X.to_pickle('reduce_x.pkl')
