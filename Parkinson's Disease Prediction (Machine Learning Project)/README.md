# Parkinson's Disease Prediction (Machine Learning Project)

This project aims to predict Parkinson's Disease Prediction using machine learning algorithms.

## Features

- Uses a xgboost model for prediction.
- Trains on a dataset to predict Parkinson's Disease based on data.
- The Model achieved accuracy up to "94%"

## Files Structure

- 'training_model.py'     : Code to train the model.
- 'prediction_code.py'    : Code to predict new data.
- 'data/'                 : Contains the datasets for training and testing.
- 'model.pkl/'            : Contains the trained model(`model.pkl`) 
- 'StandardScaler.pkl/'   : the StandardScaler to scaler testing_data(`StandareScaler.pkl`).

## Attribute Information:

Matrix column entries (attributes):
name - ASCII subject name and recording number
MDVP:Fo(Hz) - Average vocal fundamental frequency
MDVP:Fhi(Hz) - Maximum vocal fundamental frequency
MDVP:Flo(Hz) - Minimum vocal fundamental frequency
MDVP:Jitter(%),MDVP:Jitter(Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP - Several 
measures of variation in fundamental frequency
MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA - Several measures of variation in amplitude
NHR,HNR - Two measures of ratio of noise to tonal components in the voice
status - Health status of the subject (one) - Parkinson's, (zero) - healthy
RPDE,D2 - Two nonlinear dynamical complexity measures
DFA - Signal fractal scaling exponent
spread1,spread2,PPE - Three nonlinear measures of fundamental frequency variation 
