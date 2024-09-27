import log
import joblib
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import log
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from imblearn.over_sampling import SMOTE
from gensim.models.doc2vec import Doc2Vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import time
from joblib import parallel_backend
import sys
from sklearn.decomposition import SparsePCA

def splitTrainTestVal(filename, prefix):
    log.log("Reading data from CSV ...")

    df = pd.read_csv(filename, index_col=False)
    df = df = df[df['label'].notna()]
    print(
        "\n\nNOTE: There must be a 'label' column, "
        "it will be removed from the dataset before training."
    )
    # print("Dropping columns ..")
    # df = df.drop(columns=["SHA256", "label", "category"])
    print("Data Label Counts:")
    print(df["label"].value_counts())
    delete = True
    while delete:
        print(df.head())
        convert = input(
            "Do you want to "
            + log.Color.FAIL
            + "Remove "
            + log.Color.END
            + "any columns? (Y/N)?: "
        )
        if convert == "Y" or convert == "y":
            column = 0
            for c in df.columns:
                print(str(column) + ": " + c)
                column += 1
            column = input(
                "Select column(s) to delete (e.g. 1) or "
                "multiple (e.g. 1,2,3): "
            )
            if column != "":
                i = 0
                for c in df.columns:
                    if str(i) in column.split(","):
                        df = df.drop(columns=[c])
                    i += 1
        else:
            delete = False
    encode = True
    while encode:
        print(df.head())
        convert = input(
            "Do you want to "
            + log.Color.CYAN
            + "Label Encode "
            + log.Color.END
            + "any columns? (Y/N)?: "
        )
        if convert == "Y" or convert == "y":
            column = 0
            for c in df.columns:
                print(str(column) + ": " + c)
                column += 1
            column = input(
                "Select a column(s) (e.g. 1) or "
                "multiple (e.g. 1,2,3): ")
            if column != "":
                i = 0
                for c in df.columns:
                    if str(i) in column.split(","):
                        le = LabelEncoder()
                        df[c] = le.fit_transform(df[c])
                    i += 1
        else:
            encode = False

    fill = True
    while fill:
        print(df.head())
        convert = input(
            "Do you want to "
            + log.Color.CYAN
            + "Fill Blanks with 0 on "
            + log.Color.END
            + "any columns? (Y/N)?: "
        )
        if convert == "Y" or convert == "y":
            column = 0
            for c in df.columns:
                print(str(column) + ": " + c)
                column += 1
            column = input(
                "Select a column(s) (e.g. 1) or "
                "multiple (e.g. 1,2,3) or 'a' for all: ")
            if column != "":
                i = 0
                for c in df.columns:
                    if str(i) in column.split(",") or column == "a":
                        df[c] = df[c].fillna(0)
                    i += 1
        else:
            fill = False

    smote = False
    class_weight = False
    convert = input(
        "Do you want to "
        + log.Color.RED + log.Color.UNDERLINE
        + "RE-BALANCE "
        + log.Color.END
        + "the dataset? (Y/N)?: "
    )
    if convert == "Y" or convert == "y":
        print("0: SMOTE \n1: Class Weight")
        selection = input("Select an option (e.g. 1): ")
        if selection == "0":
            smote = True
        if selection == "1":
            class_weight = True

    labels = df["label"]
    df = df.drop(columns=["label"])
    features = df.to_numpy()

    label_weights = []
    if class_weight: 
        log.log("Performing class weighting ...")
        label_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
        log.log("Class weights: " + str(label_weights))
        prefix += "CW"

    if smote:
        log.log("Performing SMOTE ...")
        smoteObj = SMOTE()
        features, labels = smoteObj.fit_resample(features, labels)
        prefix += "SMOTE"
        stats(features, labels)

    convert = input(
        "Do you want to "
        + log.Color.GREEN
        + "SAVE "
        + log.Color.END
        + "this new dataset as CSV? (Y/N)?: "
    )
    if convert == "Y" or convert == "y":
        ndf = pd.DataFrame(data=features, columns=df.columns.tolist())
        ndf["label"] = list(labels)
        fname = "Datasets/"+prefix+"_MOD_Dataset.csv" 
        ndf.to_csv(fname, index=False)
        log.log("Saved to: " + fname)

    nlp = input(
        "Do you want to perform "
        + log.Color.RED + log.Color.UNDERLINE
        + "NLP "
        + log.Color.END
        + "on a field on the dataset? (Y/N)?: "
    )
    if nlp == "Y" or nlp == "y":
        column = 0
        for c in df.columns:
            print(str(column) + ": " + c)
            column += 1
        column = input(
            "Select a column (e.g. 1) to perform NLP: ")
        if column != "":
            i = 0
            print("0: TF-IDF \n1: Bag of Words \n2: Doc2Vec")
            selection = input("Select an option (e.g. 1): ")
            for c in df.columns:
                if str(i) in column.split(",") or column == "a":
                    df[c] = df[c].str.replace('\n', ' ')
                    if selection == "0":
                        prefix += "TFIDF"
                        tf = TF_IDF(df[c]).toarray()
                        spca = SparsePCA()
                        features = spca.fit_transform(tf)
                    if selection == "1":
                        prefix += "BOW"
                        bw = BagOfWords(df[c])
                        spca = SparsePCA().toarray()
                        features = spca.fit_transform(bw)
                    if selection == "2":
                        from gensim.models.doc2vec import TaggedDocument
                        prefix += "DOC2VEC"
                        preDoc = Doc_to_Vec([TaggedDocument(doc, [i]) for i, doc in enumerate(df[c])])
                        features = [preDoc.infer_vector(x.split()) for x in df[c]]
                i += 1
            # df["label"] = list(labels)
            # fname = "Datasets/"+prefix+"_MOD_Dataset.csv"
            # df.to_csv(fname, index=False)
            # log.log("Saved to: " + fname)

    log.log("Splitting data for training 60%")
    train, test, train_labels, lab = train_test_split(
        features, labels, test_size=0.4, random_state=42, stratify=labels
    )
    log.log("Splitting data for testing and validation 20/20")
    validation, test_data, validation_labels, test_labels = train_test_split(
        test, lab, test_size=0.5, random_state=42, stratify=lab
    )

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
    if not os.path.exists("CW"):
        os.mkdir("CW")

    log.log("\n\nSaving Data ...\n\n")

    joblib.dump(
        train, os.path.join("DATA/Train/", prefix + "_features.pkl"))
    joblib.dump(
        train_labels, os.path.join("DATA/Train/", prefix + "_labels.pkl"))
    joblib.dump(
        test_data, os.path.join("DATA/Test/", prefix + "_features.pkl"))
    joblib.dump(
        test_labels, os.path.join("DATA/Test/", prefix + "_labels.pkl"))
    joblib.dump(
        validation, os.path.join("DATA/Validation/", prefix + "_features.pkl"))
    joblib.dump(
        validation_labels,
        os.path.join("DATA/Validation/", prefix + "_labels.pkl")
    )

    if class_weight:
        joblib.dump(
            label_weights,
            os.path.join("DATA/Validation/", prefix + "_weights.pkl")
        )


    log.log("Done splitting data!")
    return prefix

def stats(data, labels):
    d = {
        "len": data.shape[0],
        "features": data.shape[1],
        "count": labels.value_counts(),
    }
    return d


def TF_IDF(train_docs):
    log.log("Processing TF-IDF ...")
    X = None
    with parallel_backend('threading', n_jobs=os.cpu_count()):
        start_time = time.time()
        # pipe = Pipeline([('count', CountVectorizer()), ('tfid', TfidfTransformer())]).fit(train_docs)
        # X = pipe.fit_transform(train_docs)
        vectorizer = TfidfVectorizer(token_pattern=r"\S{2,}")
        X = vectorizer.fit_transform(train_docs)
        log.log("Fit time for TF-IDF: " + str((time.time() - start_time) / 60) + " min")
    return X


def BagOfWords(train_docs):
    log.log("Processing Bag-ofWords ...")
    start_time = time.time()
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(train_docs)
    log.log("Fit time for Bag-of-Words: " + str((time.time() - start_time) / 60) + " min")
    return X


def Doc_to_Vec(train_docs):
    log.log("Processing Doc2Vec ...")
    start_time = time.time()
    model = Doc2Vec(train_docs, vector_size=1000, window=50, min_count=1, dm=0, workers=os.cpu_count())
    model.build_vocab(train_docs)
    model.train(train_docs, total_examples=model.corpus_count, epochs=model.epochs)
    log.log("Fit time for Doc2Vec: " + str((time.time() - start_time) / 60) + " min")
    return model
