import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train_model():
    df = pd.read_csv('data/products_200k.csv')
    categorical = ['category', 'brand']
    numerical = ['weight_kg', 'rating', 'warranty_years', 'power_usage_watts', 'feature_score']
    target = 'price'

    encoder = OneHotEncoder(sparse_output=False)
    encoded_cat = encoder.fit_transform(df[categorical])
    scaler = StandardScaler()
    scaled_num = scaler.fit_transform(df[numerical])

    X = np.hstack([encoded_cat, scaled_num])
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=1)

    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    joblib.dump((model, encoder, scaler), 'model/sklearn_price_model.pkl')
    return score

def predict_price(data_dict):
    model, encoder, scaler = joblib.load('model/sklearn_price_model.pkl')

    cat_features = [[data_dict['category'], data_dict['brand']]]
    num_features = [[
        data_dict['weight_kg'],
        data_dict['rating'],
        data_dict['warranty_years'],
        data_dict['power_usage_watts'],
        data_dict['feature_score']
    ]]
    encoded = encoder.transform(cat_features)
    scaled = scaler.transform(num_features)
    input_data = np.hstack([encoded, scaled])
    prediction = model.predict(input_data)
    return round(float(prediction[0]), 2)
