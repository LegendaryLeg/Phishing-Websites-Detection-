#Proof of Concept - Detecting Phishing Websites Using Machine Learning  
#COMP3260_01 - Computer Network Security
#Rakhat Bektas(t00686769) & Raiyan Mokhammad (T00668359)  

#Step 1: Import all necessary functions
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import requests

#Step 2: Train the Machine Learning model using dataset from kaggle. 
#The path on your computer should be different, just copy the path from the folder
data = pd.read_csv("C:\\Users\\m_ray\\OneDrive\\Документы\\Raiyan\\Fall_2024\\COMP_3260\\Proof Of Concept\\Data\\phishing_url_website.csv\\phishing_url_website.csv")

d = {1:'Legitimate', 0: 'Phishing'} #change names of the labels in the table for more clear output

data['label'] = data['label'].map(d) #maps new labels to dataset

features = ['IsHTTPS', 'HasDescription', 'HasSocialNet', 'HasCopyrightInfo', 'NoOfImage', 'NoOfJS']

x = data[features] #features for trainig model on
y = data['label'] #target feature

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=8)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",accuracy_score(y_test, y_pred))
print("Confusion Matrix:",conf_matrix)
print('\nClassification Report:')
print(classification_rep)

# Save the model as a .pkl file
with open('DT_phishing_model.pkl', 'wb') as file:
    pickle.dump(clf, file)

#Step 3: import trained model and evaluate it
# Load the model from the .pkl file
with open('DT_phishing_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

#Now we need to check the model on two different URLs, one will be legitemate and second phishing
#We also need to retrive required data from the URLs
def analyze_html(html_content, url):
    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract information
    is_https = 1 if urlparse(url).scheme == 'https' else 0
    
    # Count number of images
    no_of_images = len(soup.find_all('img'))
    
    # Count number of JavaScript references
    no_of_js = len(soup.find_all('script'))

    # Check for description meta tag
    has_description = 1 if soup.find('meta', attrs={'name': 'description'}) else 0

    # Check for social network links
    social_networks = ['facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com']
    has_social_net = 1 if any(social in link.get('href', '') for link in soup.find_all('a') for social in social_networks) else 0

    # Check for copyright information
    has_copyright_info = 1 if 'copyright' in soup.text.lower() or '©' in soup.text else 0

    # Create a dictionary to hold the results and returns it
    result = {
        'IsHTTPS': is_https,
        'HasDescription': has_description,
        'HasSocialNet': has_social_net,
        'HasCopyrightInfo': has_copyright_info,
        'NoOfImage': no_of_images,
        'NoOfJS': no_of_js
    }
    return result

#Reshape the dictionary to suit our model for evaluation
def covert_dict(array_features):
    return [array_features[key] for key in array_features]

#This function retrieves html code of the website 
def get_html(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None
    
url1 = "https://www.youtube.com/" #legit
url2 = "http://www.verkaufsfunnel.com"  #phishing

legit_html = get_html(url1)
phishing_html = get_html(url2)

legit_result = analyze_html(legit_html, url1)
phishing_result = analyze_html(phishing_html, url2)

legit_features = covert_dict(legit_result)
phishing_features = covert_dict(phishing_result)

print("Features of URL 1:", legit_result)
print("Features of URL 2:", phishing_result)

# Create DataFrames, this is the only format the model can take to predict values
legit_df = np.array(legit_features).reshape(1, -1)
phishing_df = np.array(phishing_features).reshape(1, -1)

#Pricts if the url is phishing or legitimate
predict_url1 = loaded_model.predict(legit_df)
predict_url2 = loaded_model.predict(phishing_df)

#Print results
print("URL 1 is:", predict_url1)
print("URL 2 is:", predict_url2)