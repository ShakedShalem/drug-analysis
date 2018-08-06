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
            'DEPNDMRJ', 'DPILLALC', 'ABODILAL', 'TOBFLAG', 'SNFFLAG', 'ALCFLAG', 'CRKFLAG', 'SUMFLAG'
            'EDUCCAT2', 'NEWRACE2', 'CATAG2', 'IRSEX', 'INCOME']]
    return X


def create_target(df):
    y = df[['AMDEYR']]
    return y


def clean_data(X, y):
    # Create a mask to remove rows 'Aged 12-17'
    mask = y[y.AMDEYR < 0]
    rows_to_remove = list(mask.index.values)
    X = X.drop(X.index[rows_to_remove])

    # Reshape y
    y = y[y.AMDEYR > 0]
    y = y.T.squeeze()
    return X, y


if __name__ == '__main__':
    df = pd.read_csv('data/druguse.tsv', sep='\t')
    X = create_features_df(df)
    y = create_target(df)
    X, y = clean_data(X, y)
