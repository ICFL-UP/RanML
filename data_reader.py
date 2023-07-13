import log 
import joblib
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder



def splitTrainTestVal(filename):
    log.log("Reading data from CSV ...")
    prefix = filename[0:3]
    df = pd.read_csv(filename, index_col="Unnamed: 0")
    df = df.dropna()
    labels = df["label"]
    df = df[['name','entropy']]
    le = LabelEncoder()
    df['name'] = le.fit_transform(df['name'])
    features = df.to_numpy()
    
    
    train, test, train_labels, lab = train_test_split(features, labels, test_size=0.4, random_state=42, stratify=labels)
    validation, test_data, validation_labels, test_labels = train_test_split(test, lab, test_size=0.5, random_state=42, stratify=lab)

    if not os.path.exists("DATA"):
        os.mkdir("DATA")
    if not os.path.exists("DATA/Train"):
        os.mkdir("DATA/Train")
    if not os.path.exists("DATA/Test"):
        os.mkdir("DATA/Test")
    if not os.path.exists("DATA/Validation"):
        os.mkdir("DATA/Validation")
    if not os.path.exists("Models"):
        os.mkdir("Models")

    log.log("\n\nSaving Data ...\n\n")
    joblib.dump(train, "DATA/Train/"+prefix+"_features.pkl")
    joblib.dump(train_labels, "DATA/Train/"+prefix+"_labels.pkl")
    joblib.dump(test_data, "DATA/Test/"+prefix+"_features.pkl")
    joblib.dump(test_labels, "DATA/Test/"+prefix+"_labels.pkl")
    joblib.dump(validation, "DATA/Validation/"+prefix+"_features.pkl")
    joblib.dump(validation_labels, "DATA/Validation/"+prefix+"_labels.pkl")

    log.log("Done splitting data!") 


def stats(data, labels):
    d = {
        'len': data.shape[0],
        'features': data.shape[1],
        'count': labels.value_counts()
    }
    return d
