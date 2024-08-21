import log
import joblib
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import log


def splitTrainTestVal(filename, prefix):
    log.log("Reading data from CSV ...")

    df = pd.read_csv(filename, index_col="Unnamed: 0")
    df = df = df[df['label'].notna()]
    print(
        "\n\nNOTE: There must be a 'label' column, "
        "it will be removed from the dataset before training."
    )
    # print("Dropping columns ..")
    # df = df.drop(columns=["SHA256", "label", "category"])
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
                        df[c].fillna(0, inplace=True)
                    i += 1
        else:
            fill = False

    labels = df["label"]
    df = df.drop(columns=["label"])
    features = df.to_numpy()

    train, test, train_labels, lab = train_test_split(
        features, labels, test_size=0.4, random_state=42, stratify=labels
    )
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

    log.log("Done splitting data!")


def stats(data, labels):
    d = {
        "len": data.shape[0],
        "features": data.shape[1],
        "count": labels.value_counts(),
    }
    return d
