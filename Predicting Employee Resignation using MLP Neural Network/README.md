# Predicting Employee Resignation using MLP Neural Network

This project aims to predict employee resignation using a Multi-Layer Perceptron (MLP) neural network model. The model is trained on employee data to identify patterns that indicate the likelihood of resignation.

## Project Overview

Employee retention is a key challenge for many organizations. This project leverages machine learning to predict employee resignations, allowing companies to proactively address retention risks. Using a neural network model (MLP), the project predicts the likelihood of an employee resigning based on various features in the dataset.

## Technologies Used

- Python
- Keras & TensorFlow
- Scikit-Learn
- Pandas & NumPy
- Matplotlib (for visualization)

## Folder contents

Model.h5: The trained file for the multilayer neural model.

StandardScaler.pkl: The scale file used to standardize data.

Train_dataset.csv: Training dataset.

test_dataset.csv: Test data set.

Train_model_code.ipynb: The training code for the model.

Predict_code.ipynb: Prediction code using the trained model.


## Operating steps

1. Train the model

Open the train_model_code.ipynb file in Google Colab or in Jupyter Notebook, and make sure the data and model files are uploaded to your environment.
Follow the steps to train and save the model.

2. Prediction

Open the predict_code.ipynb file and load the saved model (Model.h5) and the StandardScaler file (StandardScaler.pkl).
Then upload the test dataset (test_dataset.csv) to perform the prediction process


## note

This project was written in Google Colab, 
so if you want to run it in another environment like Jupyter Notebook on your machine, 
make sure all path(s) to the files are updated properly.

---

## Additional comments

You may need to modify the file paths if you upload the folder to a different environment.

For the codes to run correctly, make sure to approve the approvals used from the libraries
