# Phishing-Websites-Detection-
Phishing attacks are a growing threat, using deceptive tactics to trick users into revealing sensitive information. This project focuses on exploring differences between legitimate and forged websites and how to detect the phishing ones using machine learning. 

# Table Of Contents 
- Project Overview
- Dataset
- Technologies Used
- Algorithm
- Results 

# Project Overview 
Our project uses supervised machine learning techniques to classify websites as either phishing or legitimate. Extracting patterns from URLs and HTML content builds a predictive model capable of flagging suspicious sites before users fall victim.

# Dataset
We used the dataset from Kaggle. It consists of phishing and legitimate websites with different features: URL, Domain, IsHTTPS, LineOfCode, HasSubmitButton, and a target feature 'label' with 1 representing Legitimate and 0 as Phishing. 

Feature Engineering and Extraction 
- Preprocessing 
- Choose features (correlation matrix)

# Technologies Used
- VS Code
- Any Browser 

# Algorithm
Choose any algorithm for ML. In this project, we have tested Decision Tree and Logistic Regression to identify which algorithm performs the best with higher accuracy. 

Steps to create the ML model:
- Load dataset
- Choose features as a target (label) and for training
- Divide the data into train-test portions
- Verify how the model performs using the confusion matrix and accuracy
- Import the model using the 'pickle' library 

# Results 
In order to test the model, choose any URL and insert it into the Python file. Load the model and perform the feature extraction from the URL for a model to make a prediction.     
![image](https://github.com/user-attachments/assets/117d14f8-aca6-4f58-9a9f-68a1d76e1d62)
![image](https://github.com/user-attachments/assets/731fd452-539f-4318-b865-0bc256fe46ce)
![image](https://github.com/user-attachments/assets/cdcd3350-f724-4f1b-a099-fd204559b794)

Collaborators:
Raiyan Mokhammad - raiyan.mokhd@gmail.com
Rakhat Bektas - rakhatbektas@gmail.com
