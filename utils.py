import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

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

def read_data(path, test=False, encoder=None):
    # Read dataset into pandas dataframe
    df = pd.read_csv(path, encoding='latin1')
    # We don't need ID and don't want to train on our results
    #Extract dates for later use
    #One-hotting them will lead to loads of unnecessary feature
    IDs = df['Unnamed: 0'].values
    dates = df.iloc[:, 2].values
    data_to_onehot = pd.DataFrame(df, columns=['Strassenklasse', 'Unfallklasse', 'Lichtverhältnisse', 'Bodenbeschaffenheit', 'Geschlecht', 'Fahrzeugtyp', 'Wetterlage']).values
    non_onehot = pd.DataFrame(df, columns=['Alter', 'Verletzte Personen', 'Zeit (24h)', 'Anzahl Fahrzeuge']).values
    if not test:
        #One-hot categorical data (except dates)
        #Requires sklearn 0.20+
        if not encoder:
            encoder = OneHotEncoder(categories='auto')
            one_hotted = encoder.fit_transform(data_to_onehot).todense()
        else:
            one_hotted = encoder.transform(data_to_onehot).todense()
        # Result vector
        y = df['Unfallschwere'].values
    else:
        #No results in test data
        one_hotted = encoder.transform(data_to_onehot).todense()
        y = 0
    #fix the dates
    dates = fix_dates(dates)[:, None]
    X = np.concatenate((dates, non_onehot, one_hotted), axis=1)
    return IDs, X, y, encoder

def split_data(X, y):
    # Split so we have validation data to score our performance
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15)

    return X_tr, X_val, y_tr, y_val
