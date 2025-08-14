#########################################################################################
# Required Python packages
#########################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

#########################################################################################
# File Paths
#########################################################################################
INPUT_PATH = "breast-cancer-wisconsin.data"
OUTPUT_PATH = "breast-cancer-wisconsin.csv"
MODEL_PATH = "bc_rf_pipeline.joblib"

#########################################################################################
# Headers
#########################################################################################

HEADERS = ["CodeNumber","ClumpThickness","UniformityCellSize","UniformityCellShape",
           "MarginalAdhesion","SingleEpithelialCellSize","BareNuclei","BlandChromatin","NormalNucleoli",
           "Mitoses","CancerType"]

#########################################################################################
# Function Name : get_headers
# Description : Read the data into pandas dataframe
# Input : path of csv file
# output : Gives the data 
# Author : Sohan Satish Suralkar
# Data : 14/08/2025
#########################################################################################

def read_data(path):
    """Read the data into pandas dataframe"""
    data = pd.read_csv(path,header=None)
    return data

#########################################################################################
# Function Name : get_headers
# Description : Dataset headers
# Input : dataset
# output : Returns the headers 
# Author : Sohan Satish Suralkar
# Data : 14/08/2025
#########################################################################################

def get_headers(dataset):
    """Returns the dataset headers"""
    return dataset.columns.values

#########################################################################################
# Function Name : add_headers
# Description : Add the headers to the dataset
# Input : dataset
# output : Returns the updates dataset
# Author : Sohan Satish Suralkar
# Data : 14/08/2025
#########################################################################################

def add_headers(dataset,headers):
    """Add dataset headers"""
    dataset.columns = headers
    return dataset

#########################################################################################
# Function Name : data_file_to_csv
# Description : Convert the raw .data file to csv file Headers
# Input : Nothing
# output : Write the data to csv
# Author : Sohan Satish Suralkar
# Data : 14/08/2025
#########################################################################################

def data_file_to_csv():
    """Convert the raw .data file to csv file Headers"""
    dataset = read_data(INPUT_PATH)
    dataset = add_headers(dataset,HEADERS)
    dataset.to_csv(OUTPUT_PATH,index=False)
    print("File saved...!")

#########################################################################################
# Function Name : handle_missing_values
# Description : filter missing values from the dataset
# Input : Dataset With missing values
# output : Dataset by remocing missing values
# Author : Sohan Satish Suralkar
# Data : 14/08/2025
#########################################################################################

def handle_missing_values(df,features_headers):
    """Convert "?" to NaN and let SimpleImputer handle them inside PipeLine.
    Keep Only numerical columns in features."""
    # Replace "?" in the whole dataframe
    df = df.replace("?",np.nan)

    # Cast Features to numeric
    df[features_headers] = df[features_headers].apply(pd.to_numeric,errors="coerce")

    return df

#########################################################################################
# Function Name : split_dataset
# Description : split dataset with train_percentage
# Input : Dataset With related information
# output : Dataset after spliting
# Author : Sohan Satish Suralkar
# Data : 14/08/2025
#########################################################################################

def split_dataset(dataset,train_percentage,feature_headers,target_headers,random_state = 42):
    """Split dataset into train/test"""
    train_x,test_x,train_y,test_y = train_test_split(
        dataset[feature_headers],dataset[target_headers],
        train_size=train_percentage,random_state=random_state,stratify=dataset[target_headers]
        )

    return train_x,test_x,train_y,test_y

#########################################################################################
# Function Name : dataset_statistics
# Description : Display the statistics
# Author : Sohan Satish Suralkar
# Data : 14/08/2025
#########################################################################################

def dataset_statistics(dataset):
    """Print Basis stats"""
    print(dataset.describe(include="all"))

#########################################################################################
# Function Name : build_pipeline
# Description : Build the pipeline
# SimpleImputer : Replace missing with median
#                   RandomForestClassifier : robust baseline 
# Author : Sohan Satish Suralkar
# Data : 14/08/2025
#########################################################################################

def build_pipeline():
    pipe = Pipeline(steps=[("imputer",SimpleImputer(strategy="median")),
                           ("rf",RandomForestClassifier(
                               n_estimators=300,
                               random_state=42,
                               n_jobs=-1,
                               class_weight=None
                           ))
                        ])
    return pipe

#########################################################################################
# Function Name : train_pipeline
# Description : Train the pipeline
# Author : Sohan Satish Suralkar
# Data : 14/08/2025
#########################################################################################

def train_pipeline(pipeline,x_train,y_train):
    pipeline.fit(x_train,y_train)
    return pipeline

#########################################################################################
# Function Name : save_model
# Description : Save the model
# Author : Sohan Satish Suralkar
# Data : 14/08/2025
#########################################################################################

def save_model(model,path=MODEL_PATH):
    joblib.dump(model,path)
    print(f"Model saved to {path}")

#########################################################################################
# Function Name : load_model
# Description : load the  trained model
# Author : Sohan Satish Suralkar
# Data : 14/08/2025
#########################################################################################

def load_model(path=MODEL_PATH):
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model

#########################################################################################
# Function Name : plot_confusion_matrix_matshow
# Description : Display Confusion Matrix
# Author : Sohan Satish Suralkar
# Data : 14/08/2025
#########################################################################################

def plot_confusion_matrix_matshow(y_true,y_pre, title = "Confusion Matrix"):
    cm = confusion_matrix(y_true,y_pre)
    fig, ax = plt.subplots()
    cax = ax.matshow(cm)
    fig.colorbar(cax)
    for (i,j) , v in np.ndenumerate(cm):
        ax.text(j,i,str(v),ha = "center",va = "center")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

#########################################################################################
# Function Name : plot_feature_importances
# Description : Display the feature importance
# Author : Sohan Satish Suralkar
# Data : 14/08/2025
#########################################################################################

def plot_feature_importance(model,feature_names,title = "Feature Importances (Random Forest)"):
    if hasattr(model,"named_steps") and "rf" in model.named_steps:
        rf = model.named_steps["rf"]
        importances = rf.feature_importances_
    elif hasattr(model,"feature_importances_"):
        importances = model.feature_importances_
    else :
        print("feature Importance not available for this model .. ")

    idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(8,4))
    plt.bar(range(len(importances)),importances[idx])
    plt.xticks(range(len(importances)),[feature_names[i]for i in idx], rotation = 45, ha = "right")
    plt.ylabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.show()

#########################################################################################
# Function Name : main
# Description : Main Functions Where the execution starts
# Author : Sohan Satish Suralkar
# Data : 14/08/2025
#########################################################################################

def main():
    #1. Ensure CSV exists (run once if needed)
    #data_file_to_csv()

    #2. Load CSV
    dataset = pd.read_csv(OUTPUT_PATH)

    #3. basis Stats
    dataset_statistics(dataset)

    #4. Prepare features /target
    feature_headers = HEADERS[1:-1]  # drop CodeNumber keel all features
    target_headers = HEADERS[-1]   # Cancer Type (2 = benign , 4 = malignant)

    #5. Handle "?" and coerce to numeric : imputation will happen inside pipeline
    dataset = handle_missing_values(dataset,feature_headers)
    #6. Split
    train_x, test_x, train_y,test_y = split_dataset(dataset,0.7,feature_headers,target_headers)

    print("Train_x Shape :: ",train_x.shape)
    print("Train_y Shape :: ",train_y.shape)
    print("Test_x Shape :: ",test_x.shape)
    print("Test_y Shape :: ",test_y.shape)

    #7. Build + Train Pipeline
    Pipeline = build_pipeline()
    trained_model = train_pipeline(Pipeline,train_x,train_y)
    print("Trained Pipeline :: ",trained_model)

    #8. Predictions
    predictions = trained_model.predict(test_x)

    #9. Metrics
    print("Train Accuracy :: ", accuracy_score(train_y,trained_model.predict(train_x)))
    print("Test Accuracy :: ", accuracy_score(test_y,predictions))
    print("Classification Report :\n", classification_report(test_y,predictions))
    print("Confusion Matrix : \n",confusion_matrix(test_y,predictions))

    # Feature Importamce Tree based 
    plot_feature_importance(trained_model,feature_headers, title="feature Importances (RF)")

    # 10 . Save The model (pipeline )using Joblib
    save_model(trained_model,MODEL_PATH)
    # 11.  Load Model And testt a sample
    loaded = load_model(MODEL_PATH)
    sample = test_x.iloc[[0]]
    pred_loaded = loaded.predict(sample)
    print(f"Loaded  Model Prediction for first sample : {pred_loaded[0]}")

#########################################################################################
# Application Starter
#########################################################################################

if __name__ == "__main__":
    main()
