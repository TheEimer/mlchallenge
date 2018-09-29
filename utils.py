import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def fix_dates(dates):
    # For each entry we check the month
    for s, i in zip(dates, np.arange(len(dates))):
        if "Jan" in s:
            dates[i] = 1
        elif "Feb" in s:
            dates[i] = 2
        elif "Mar" in s:
            dates[i] = 3
        elif "Apr" in s:
            dates[i] = 4
        elif "May" or "Mai" in s:
            dates[i] = 5
        elif "Jun" in s:
            dates[i] = 6
        elif "Jul" in s:
            dates[i] = 7
        elif "Aug" in s:
            dates[i] = 8
        elif "Sep" in s:
            dates[i] = 9
        elif "Oct" in s:
            dates[i] = 10
        elif "Nov" in s:
            dates[i] = 11
        elif "Dec" in s:
            dates[i] = 12
        # If we can't match, we default to zero
        else:
            dates[i] = 0

    return dates

def read_data(path, test=False):
    # Read dataset into pandas dataframe
    df = pd.read_csv(path, encoding='latin1')
    # We don't need ID and don't want to train on our results
    IDs = df['Unnamed: 0'].values
    X = df.iloc[:, 1:].drop('Unfallschwere', axis=1)
    # Result vector
    if not test:
        y = df['Unfallschwere'].values
    else:
        y = 0
    # One-hot the data and fix the dates
    X = pd.get_dummies(X, columns=['Strassenklasse', 'Unfallklasse', 'Lichtverhältnisse', 'Bodenbeschaffenheit', 'Geschlecht', 'Fahrzeugtyp', 'Wetterlage']).values
    X[:, 0] = fix_dates(X[:, 0])
    return IDs, X, y

def split_data(X, y):
    # Split so we have validation data to score our performance
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15)

    return X_tr, X_val, y_tr, y_val
