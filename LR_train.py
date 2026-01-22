import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# import dataset
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
#print(df.head(2))

# data preprocessing 

df = df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1)
# target and feature
x = df.drop('Attrition', axis=1)
y = df['Attrition']
# attrition column encoding without map
le =LabelEncoder()
y = le.fit_transform(y)

# find categorical and numerical columns

cat_cols = x.select_dtypes(include=['object']).columns
nums_cols = x.select_dtypes(include=['int64', 'float64']).columns
print(nums_cols)
# preprocessing pipelines for both numerical and categorical data

nums_tramsformer = Pipeline(
    steps =[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)

cat_transformer = Pipeline(
    steps = [
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('endcoder', OneHotEncoder(handle_unknown ='ignore'))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', nums_tramsformer, nums_cols),
        ('cat', cat_transformer, cat_cols)
    ]
)

# Logistic Regression model pipeline

log_model = LogisticRegression(max_iter=500)

# full pipeline

log_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('logistic_regression', log_model)
    ]
)

# evaluation function

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

log_pipeline.fit(x_train, y_train)
y_pred = log_pipeline.predict(x_test)

# evaluation metrics

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# save the model 
with open('Logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(log_pipeline, f)

print("Model saved as logistic_regression_model.pkl")


