import gradio as gr
import pandas as pd
import pickle
import numpy as np

# load the model 

with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)


# the logic function

def predict_attrition(
        Age, DistanceFromHome,MonthlyIncome,JobSatisfaction,NumCompaniesWorked,TotalWorkingYears,
        YearsAtCompany,YearsInCurrentRole,YearsSinceLastPromotion,PercentSalaryHike,YearsWithCurrManager,
        BusinessTravel,Department,EducationField,JobRole,OverTime,
        JobLevel,PerformanceRating,DailyRate,MonthlyRate,Education,HourlyRate,
RelationshipSatisfaction,Gender,StockOptionLevel,WorkLifeBalance,
JobInvolvement,TrainingTimesLastYear,MaritalStatus,EnvironmentSatisfaction):
    input_df = pd.DataFrame([[
        Age, DistanceFromHome,MonthlyIncome,JobSatisfaction,NumCompaniesWorked,TotalWorkingYears,
        YearsAtCompany,YearsInCurrentRole,YearsSinceLastPromotion,PercentSalaryHike,YearsWithCurrManager,
        BusinessTravel,Department,EducationField,JobRole,OverTime,JobLevel,PerformanceRating,DailyRate,MonthlyRate,Education,HourlyRate,
RelationshipSatisfaction,Gender,StockOptionLevel,WorkLifeBalance,
JobInvolvement,TrainingTimesLastYear,MaritalStatus,EnvironmentSatisfaction
    ]], 
    columns = [
        'Age', 'DistanceFromHome','MonthlyIncome','JobSatisfaction','NumCompaniesWorked','TotalWorkingYears',
        'YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','PercentSalaryHike','YearsWithCurrManager',
        'BusinessTravel','Department','EducationField','JobRole','OverTime',
        'JobLevel','PerformanceRating','DailyRate','MonthlyRate','Education','HourlyRate',
        'RelationshipSatisfaction','Gender','StockOptionLevel','WorkLifeBalance',
        'JobInvolvement','TrainingTimesLastYear','MaritalStatus','EnvironmentSatisfaction'
    ])

    # prediction

    Prediction = model.predict(input_df)[0]

    # return formatted result 
    return "Yes" if Prediction == 1 else "No"

# Gradio interface 

inputs =[ 
   
    gr.Slider(18,60, step=1, label="Age"),
    gr.Slider(0, 28,  label="DistanceFromHome"),
    gr.Slider(1000,25000, step=100, label="MonthlyIncome"),
    gr.Slider(0,5, step=1, label="JobSatisfaction" ),
    gr.Slider(0,10, step=1, label="NumCompaniesWorked"),
    gr.Slider(0,25, step=1, label="TotalWorkingYears"),
    gr.Slider(0,25, step=1, label="YearsAtCompany"),
    gr.Slider(0,10, step=1, label="YearsInCurrentRole"),
    gr.Slider(0,15, step=1, label="YearsSinceLastPromotion"),
    gr.Slider(10,25, step=1, label="PercentSalaryHike"),
    gr.Slider(0,10, step=1, label="YearsWithCurrManager"),
   
    gr.Dropdown(choices=['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'], label="BusinessTravel"),
    gr.Dropdown(choices=['Sales', 'Research & Development', 'Human Resources'], label="Department"),
    gr.Dropdown(choices=['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources' ], label="EducationField"),
    gr.Dropdown(choices=[
        'Sales Executive', 'Research Scientist', 'Laboratory Technician','Manufacturing Director',
        'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources' ], label="JobRole"),
    gr.Dropdown(choices=['Yes', 'No'], label="OverTime"),

    gr.Slider(1,5, step=1, label="JobLevel"),
    gr.Slider(1,5, step=1, label="PerformanceRating"),
    gr.Slider(100,1500, step=10, label="DailyRate"),
    gr.Slider(1000,20000, step=100, label="MonthlyRate"),
    gr.Slider(1,5, step=1, label="Eductaion"),
    gr.Slider(20,100, step=1, label="HourlyRate"),
    gr.Slider(1,5, step=1, label="RelationshipSatisfaction"),
    gr.Dropdown(choices=['Male', 'Female'], label="Gender"),
    gr.Slider(0,4, step=1, label="StockOptionLevel"),
    gr.Slider(1,5, step=1, label="WorkLifeBalance"),
    gr.Slider(1,5, step=1, label="JobInvolvement"),
    gr.Slider(0,5, step=1, label="TrainingTimesLastYear"),
    gr.Dropdown(choices=['Single', 'Married', 'Divorced'], label="MaritalStatus"),
    gr.Slider(1,5, step=1, label="EnvironmentSatisfaction")
]

app = gr.Interface(
    fn = predict_attrition,
    inputs = inputs,
    outputs= gr.Textbox(label="Attrition Prediction"),
    title= "Employee Attrition Prediction "
) 

app.launch() 

    

   
    