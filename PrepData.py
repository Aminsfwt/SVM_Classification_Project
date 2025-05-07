#import necessary libraries
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

#read the data
def read_data(file_path, output, id):
    """
    Reads data from a CSV file and returns the DataFrame.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: DataFrame containing the data.
    """
    # Read the data from the CSV file
    data = pd.read_csv(file_path)

# Deal with missing data using median imputation
#imputer = SimpleImputer(strategy='median')
#data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Drop the 'id' column if it exists
    if id in data.columns:
        data = data.drop(columns=[id])

    #convert the target variable to binary (0 or 1)
    for i in range(data.shape[0]):
        if(data.iloc[i,-1]>0):
           data.iat[i,-1]=1
        else:
            data.iat[i,-1]=0 

    # Extract features and target variable
    # X = pd.get_dummies(X, columns=["cp", "thal", "slope"], drop_first=True)
    X = pd.get_dummies(data.drop(columns=[output]))  # Exclude output (target) and encode categoricals
    y = data[output]
    
    return X, y
    


