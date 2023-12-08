import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encoder(data, sparse_features):

    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    
    return data


def decoder(result, raw, data):

    for col in data.columns[:2]:
        lbe = LabelEncoder()
        lbe.fit(raw[col])
        result[col] = lbe.inverse_transform(data[col])
    
    return result