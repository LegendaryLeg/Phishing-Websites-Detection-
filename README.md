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

# Model Comparison Results

Holdout metrics on the **deployable** feature set (features a Chrome extension can extract from URL + DOM). Label **0 = Phishing**, **1 = Legitimate**. Phishing-class scores below are computed from the confusion matrices produced by `train_compare_models.py`.

| Model | Phishing precision | Phishing recall | Phishing F1 |
| --- | ---: | ---: | ---: |
| Decision Tree | 0.999059 | 0.999009 | 0.999034 |
| Logistic Regression | 0.999306 | 0.998762 | 0.999034 |
| Random Forest (n=200) | 1.000000 | 0.999752 | 0.999876 |
| XGBoost (n=200) | 0.999950 | 0.999901 | 0.999926 |

**Chosen model: Decision Tree** (`best_phishing_model.pkl` → `model.js` via m2cgen).

**Why:** Selection priority was (1) phishing recall, (2) phishing F1, (3) deployable JS size. XGBoost led Decision Tree by only ~0.09 points of phishing recall/F1 (well under the ~1–2 point ensemble threshold), so the single tree was preferred for a much smaller client-side bundle (~30 KB vs a large ensemble export). No `n_estimators` sweep was needed because an ensemble was not selected.

**Final deployable feature list (41 features):**  
`URLLength`, `DomainLength`, `IsDomainIP`, `TLDLength`, `NoOfSubDomain`, `HasObfuscation`, `NoOfObfuscatedChar`, `ObfuscationRatio`, `NoOfLettersInURL`, `LetterRatioInURL`, `NoOfDegitsInURL`, `DegitRatioInURL`, `NoOfEqualsInURL`, `NoOfQMarkInURL`, `NoOfAmpersandInURL`, `NoOfOtherSpecialCharsInURL`, `SpacialCharRatioInURL`, `IsHTTPS`, `LineOfCode`, `LargestLineLength`, `HasTitle`, `HasFavicon`, `IsResponsive`, `HasDescription`, `NoOfPopup`, `NoOfiFrame`, `HasExternalFormSubmit`, `HasSocialNet`, `HasSubmitButton`, `HasHiddenFields`, `HasPasswordField`, `Bank`, `Pay`, `Crypto`, `HasCopyrightInfo`, `NoOfImage`, `NoOfCSS`, `NoOfJS`, `NoOfSelfRef`, `NoOfEmptyRef`, `NoOfExternalRef`

These names/order match `FEATURE_ORDER` in `model.js` and the object returned by `extractFeaturesFromPage()` in `popup.js`. Exported `CLASS_ORDER` is `[0, 1]` (matches `clf.classes_`).
