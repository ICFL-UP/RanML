import sys
import log
from log import Color
from optparse import OptionParser
import os

### 1 = Malicious 0 = Benign

### Globals
FILE = ""
# MODEL_LIST = ['RF','GBT', 'AB', 'KNN', 'DT', 'NB', 'LR', 'XGB', 'NN', 'SVM']
MODEL_LIST = [
    "DT",
    "KNN",
    "RF",
    "NN",
    "NB",
    "SVM",
    "XGB",
    "GBT",
    "AB",
    "KM",
    "LR",
    "BAG",
]
DATA = 0
TRAIN = 0
PREDICT = 0
ROC = 0
BEST = 0
CM = 0
SMOTE = 0
CW = 0
NLP = 0


def main():
    from sklearn.metrics import RocCurveDisplay
    import joblib
    import matplotlib.pyplot as plt
    import classifiers
    import data_reader
    import pandas as pd
    from datetime import datetime
    import traceback
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import ConfusionMatrixDisplay

    prefix = os.path.basename(FILE)
    prefix = prefix[0:prefix.index("_")]
    CW_arr = []
    print(datetime.now())

    # # STATS
    data = pd.read_csv(FILE)
    if "category" in data:
        log.log(
            prefix + " Dataset stats - Category: " +
            str(data["category"].value_counts())
        )
        log.log(prefix + " Dataset stats - Label: \n" +
                str(data["label"].value_counts()))
        log.log(prefix + " Dataset stats - Shape: " + str(data.shape))

    if DATA:
        log.log("Preparing data splitting ...")
        prefix = data_reader.splitTrainTestVal(FILE, prefix)

    log.log("Loading data ..")
    if SMOTE:
        prefix += "SMOTE"
    if CW:
        try:
            CW_arr = joblib.load("DATA/Train/" + prefix + "_weights.pkl")
        except Exception:
            print("Error: No Weights found, --cw flag must have weights which are generated when -d flag is used and --cw")
            sys.exit()
    
    if NLP == 1:
        prefix += "TFIDF"
    if NLP == 2:
        prefix += "BOW"
    if NLP == 3:
        prefix += "DOC2VEC"

    X = {
        "TRAIN": joblib.load("DATA/Train/" + prefix + "_features.pkl"),
        "VAL": joblib.load("DATA/Validation/" + prefix + "_features.pkl"),
        "TEST": joblib.load("DATA/Test/" + prefix + "_features.pkl"),
    }
    Y = {
        "TRAIN": joblib.load("DATA/Train/" + prefix + "_labels.pkl"),
        "VAL": joblib.load("DATA/Validation/" + prefix + "_labels.pkl"),
        "TEST": joblib.load("DATA/Test/" + prefix + "_labels.pkl"),
    }

    log.log(
        "Dataset TRAIN {}: \n".format(prefix)
        + str(Y["TRAIN"].value_counts()))
    log.log(
        "Dataset VAL {}: \n".format(prefix)
        + str(Y["VAL"].value_counts()))
    log.log(
        "Dataset TEST {}: \n".format(prefix)
        + str(Y["TEST"].value_counts()))

    if TRAIN:
        # Classifier Training
        log.log("\n\nTraining Classifiers ...")

        if "RF" in MODEL_LIST:
            try:
                if CW:
                    classifiers.randomForrest(X["TRAIN"], Y["TRAIN"], prefix, CW=CW_arr)
                else:
                    classifiers.randomForrest(X["TRAIN"], Y["TRAIN"], prefix)

            except Exception as e:
                print(traceback.print_exc())
                log.log(e)
                log.log("\n\n\n\n\nERROR in training of RF\n\n\n\n")

        if "GBT" in MODEL_LIST:
            try:
                classifiers.gradientBoost(X["TRAIN"], Y["TRAIN"], prefix)
            except Exception as e:
                print(traceback.print_exc())
                log.log(e)
                log.log("\n\n\n\n\nERROR in training of DT\n\n\n\n")

        if "AB" in MODEL_LIST:
            try:
                classifiers.adaBoost(X["TRAIN"], Y["TRAIN"], prefix)
            except Exception as e:
                print(traceback.print_exc())
                log.log(e)
                log.log("\n\n\n\n\nERROR in training of AB\n\n\n\n")

        if "KNN" in MODEL_LIST:
            try:
                classifiers.knn(X["TRAIN"], Y["TRAIN"], prefix)
            except Exception as e:
                print(traceback.print_exc())
                log.log(e)
                log.log("\n\n\n\n\nERROR in training of KNN\n\n\n\n")

        if "DT" in MODEL_LIST:
            try:
                if CW:
                    classifiers.decisionTree(X["TRAIN"], Y["TRAIN"], prefix, CW=CW_arr)
                else:
                    classifiers.decisionTree(X["TRAIN"], Y["TRAIN"], prefix)
            except Exception as e:
                print(traceback.print_exc())
                log.log(e)
                log.log("\n\n\n\n\nERROR in training of DT\n\n\n\n")

        if "NB" in MODEL_LIST:
            try:
                classifiers.naiveBayes(X["TRAIN"], Y["TRAIN"], prefix)
            except Exception as e:
                print(traceback.print_exc())
                log.log(e)
                log.log("\n\n\n\n\nERROR in training of NB\n\n\n\n")

        if "XGB" in MODEL_LIST:
            try:
                classifiers.xgboost(X["TRAIN"], Y["TRAIN"], prefix)
            except Exception as e:
                print(traceback.print_exc())
                log.log(e)
                log.log("\n\n\n\n\nERROR in training of XGB\n\n\n\n")

        if "LR" in MODEL_LIST:
            try:
                if CW:
                    classifiers.logisticRegression(X["TRAIN"], Y["TRAIN"], prefix, CW=CW_arr)
                else:
                    classifiers.logisticRegression(X["TRAIN"], Y["TRAIN"], prefix)
            except Exception as e:
                print(traceback.print_exc())
                log.log(e)
                log.log("\n\n\n\n\nERROR in training of LR\n\n\n\n")

        if "KM" in MODEL_LIST:
            try:
                classifiers.kmeans(X["TRAIN"], Y["TRAIN"], prefix)
            except Exception:
                print(traceback.print_exc())
                log.log("\n\n\n\n\nERROR in training of KM\n\n\n\n")

        if "NN" in MODEL_LIST:
            try:
                classifiers.nn(X["TRAIN"], Y["TRAIN"], prefix)
            except Exception as e:
                print(traceback.print_exc())
                log.log(e)
                log.log("\n\n\n\n\nERROR in training of NN\n\n\n\n")

        if "SVM" in MODEL_LIST:
            try:
                if CW:
                    classifiers.svm(X["TRAIN"], Y["TRAIN"], prefix, CW=CW_arr)
                else:
                    classifiers.svm(X["TRAIN"], Y["TRAIN"], prefix)
            except Exception as e:
                print(traceback.print_exc())
                log.log(e)
                log.log("\n\n\n\n\nERROR in training of SVM\n\n\n\n")

        if "BAG" in MODEL_LIST:
            try:
                classifiers.bagging(X["TRAIN"], Y["TRAIN"], prefix)
            except Exception as e:
                print(traceback.print_exc())
                log.log(e)
                log.log("\n\n\n\n\nERROR in training of Bagging\n\n\n\n")

    # ___________________________________________________________________________
    if PREDICT:
        # Predict
        log.log("\n\nPREDICTING ...\n\n")
        results = []
        models = {}
        failed = ""
        # le = LabelEncoder()
        for mdl in MODEL_LIST:
            try:
                models[mdl + "_" + prefix] = joblib.load(
                    "Models/{}_{}_model.pkl".format(mdl, prefix)
                )

                results.append(
                    classifiers.evaluate_model(
                        mdl,
                        models[mdl + "_" + prefix],
                        X["VAL"],
                        Y["VAL"],
                        prefix,
                    )
                )
            except Exception as e:
                print(traceback.print_exc())
                log.log("\n\nFailed to load model " + mdl + " " + prefix)
                results.append([mdl]+[""]*12)
                failed += "\n\nFailed to load model " + mdl + " " + prefix
                continue
        from tabulate import tabulate
        import numpy as np
        results.insert(0, [
                    "Name",
                    "TP",
                    "TN",
                    "FP",
                    "FN",
                    "Recall",
                    "Precision",
                    "F1-Score",
                    "AUC",
                    "LogLoss",
                    "Latency(ms)",
                    "Accuracy",
                ],)
        # results = np.array(results, dtype=object).transpose()
        log.log(
            "\n\n ====== " + prefix + " ======\n\n"
            + tabulate(
                results, headers="firstrow", tablefmt='latex'
            )
            + "\n\n\n"
        )
        log.log("\n____FAILED_____\n"+failed)

    # ___________________________________________________________________________
    if ROC:
        # ROC Curve
        if not os.path.exists("ROC"):
            os.mkdir("ROC")
        print("\n\n\nGenerating ROC\n\n")
        models = {}
        for mdl in MODEL_LIST:
            try:
                models[mdl + "_" + prefix] = joblib.load(
                    "Models/{}_{}_model.pkl".format(mdl, prefix)
                )
            except Exception:
                log.log("\n\nFailed to load model " + mdl + " " + prefix)
                continue

        fig = plt.figure(figsize=(7, 7), dpi=300)
        axes = fig.gca()
        axes.set_title(prefix)
        for x in MODEL_LIST:
            try:
                RocCurveDisplay.from_estimator(
                    models[x + "_" + prefix], X["VAL"], Y["VAL"], ax=axes
                )
            except Exception:
                log.log("Failed to generate ROC for " + x)
                continue

        plt.savefig("ROC/" + prefix + "_ROC.png")

    # ___________________________________________________________________________
    if CM:
        # Confusion Matrix
        print("\n\n\nGenerating Confusion Matrix\n\n\n")
        models = {}
        for mdl in MODEL_LIST:
            try:
                models[mdl + "_" + prefix] = joblib.load(
                    "Models/{}_{}_model.pkl".format(mdl, prefix)
                )
            except Exception:
                log.log("\n\nFailed to load model " + mdl + " " + prefix)
                continue

        if not os.path.exists("CM"):
            os.mkdir("CM")
        for x in MODEL_LIST:
            try:
                fig = plt.figure(figsize=(7, 7), dpi=300)
                axes = fig.gca()
                
                ConfusionMatrixDisplay.from_estimator(
                    models[x + "_" + prefix], X["VAL"], Y["VAL"], ax=axes
                )

                plt.savefig("CM/" + x + "_" + prefix + ".png")
            except Exception:
                log.log("\n\nFailed to generate CM for " + mdl + " " + prefix)
                continue

    # ___________________________________________________________________________
    if BEST:
        # # BEST MODELS
        print("\n\n\nBEST MODELS\n\n")
        mdl = "RF"
        log.log("Training bagging on " + mdl)
        best = joblib.load("Models/{}_{}_model.pkl".format(mdl, prefix))
        classifiers.bagging_best(X["TRAIN"], Y["TRAIN"], prefix, best)

        results = classifiers.evaluate_model(
            "BEST_" + prefix, best, X["TEST"], Y["TEST"], prefix
        )
        from tabulate import tabulate

        log.log(
            tabulate(
                results,
                headers=[
                    "Name",
                    "TP",
                    "TN",
                    "FP",
                    "FN",
                    "Recall",
                    "Precision",
                    "F1-Score",
                    "AUC",
                    "LogLoss",
                    "Latency(ms)",
                    "Num",
                    "Accuracy",
                ],
            )
        )


# ___________________________________________________________________________
# ___________________________________________________________________________
# ___________________________________________________________________________


if __name__ == "__main__":
    print(
        "\n\nWelcome to "
        + Color.GREEN
        + "Ran"
        + Color.BLUE
        + "M"
        + Color.RED
        + "L\n\n"
        + Color.END
    )
    if len(sys.argv[1:]) == 0:
        print("No arguments passed, please use -h or --help for help")
        sys.exit()

    parser = OptionParser(option_class=log.MultipleOption)
    parser.add_option(
        "-d",
        "--data",
        action="store_true",
        help="Preprocess the data from the CSV file, requires -i flag",
    )
    parser.add_option(
        "-t", "--train",
        action="store_true",
        help="Train the models, can be used with -m to specify the models to train")
    parser.add_option(
        "-p",
        "--predict",
        action="store_true",
        help="Predict from the trained models, can be used with -m to specify the models to predict"
    )
    parser.add_option(
        "-r",
        "--roc",
        action="store_true",
        help="Produce the ROC Curve of the models predictions",
    )
    parser.add_option(
        "-b",
        "--best",
        action="store_true",
        help="Run the best models and compare on new data",
    )
    parser.add_option(
        "-c",
        "--confusion",
        action="store_true",
        help="Produce the Confusion matrix image for the models",
    )
    parser.add_option(
        "--smote",
        action="store_true",
        help="Use the dataset that SMOTE was performed on",
    )
    parser.add_option(
        "--cw",
        action="store_true",
        help="Use Class Weights for DT, RF, SVM, LR",
    )
    parser.add_option(
        "--tfidf",
        action="store_true",
        help="Use TFIDF training data based on input file",
    )
    parser.add_option(
        "--bow",
        action="store_true",
        help="Use Bag of Words training data based on input file",
    )
    parser.add_option(
        "--doc2vec",
        action="store_true",
        help="Use Doc2Vec training data based on input file",
    )
    parser.add_option(
        "-i", "--input", dest="input",
        help="Specify the input file")
    parser.add_option(
        "-s", "--silent", action="store_true",
        help="Silent - no prompts to verify"
    )
    parser.add_option(
        "-m",
        "--models",
        dest="models",
        metavar="MODELS",
        action="extend",
        help="Select the Models to be applied, by default all are applied,"
        "multiple can be selected comma-delimited. Valid choices are: %s."
        % (MODEL_LIST),
    )

    (options, args) = parser.parse_args()

    if options.data is not None:
        DATA = 1
    if options.train is not None:
        TRAIN = 1
    if options.smote is not None:
        SMOTE = 1
    if options.cw is not None:
        CW = 1
    if options.tfidf is not None:
        NLP = 1
    if options.bow is not None:
        NLP = 2
    if options.doc2vec is not None:
        NLP = 3
    if options.predict is not None:
        PREDICT = 1
    if options.roc is not None:
        ROC = 1
    if options.confusion is not None:
        CM = 1
    if options.best is not None:
        BEST = 1
    if options.input is None:
        print("-i flag for the Input file must be specified")
        sys.exit()
    else:
        if not options.input.endswith(".csv"):
            print("Please use a csv file for training")
            sys.exit()
        if not os.path.isfile(options.input):
            print("Error: File does not exist")
            sys.exit()
        FILE = options.input
    if options.models is not None:
        if (
            options.train is not None
            or options.predict is not None
            or options.roc is not None
        ):
            newList = []
            for m in options.models:
                if m in MODEL_LIST:
                    newList.append(m)
            MODEL_LIST = newList
        else:
            print("-m can only be used with the following flags [-t, -p, -r]")
    if (
        options.train is None
        and options.data is None
        and options.predict is None
        and options.roc is None
        and options.best is None
        and options.confusion is None
    ):
        print("No valid arguments passed, please use -h or --help for help")
        sys.exit()
    
    print("Your choices: ")
    print("Data: ", Color.GREEN + "True" + Color.END if DATA == 1 else "False")
    print(
        "Train: ", Color.GREEN + "True" +
        Color.END if TRAIN == 1 else "False")
    print(
        "SMOTE: ", Color.GREEN + "True" +
        Color.END if SMOTE == 1 else "False")
    print(
        "Class Weight: ", Color.GREEN + "True" +
        Color.END if CW == 1 else "False")
    print(
        "NLP: ", "False" if NLP == 0 else Color.GREEN + "TFIDF" if NLP == 1 else Color.GREEN + "BOW" if NLP == 2 else Color.GREEN + "DOC2VEC", Color.END)
    print(
        "Predict: ", Color.GREEN + "True" +
        Color.END if PREDICT == 1 else "False")
    print("ROC: ", Color.GREEN + "True" + Color.END if ROC == 1 else "False")
    print("CM: ", Color.GREEN + "True" + Color.END if CM == 1 else "False")
    print("BEST: ", Color.GREEN + "True" + Color.END if BEST == 1 else "False")
    print("Models: ", MODEL_LIST)

    if not options.silent:
        check = input("\nPlease verify your choices (Y/N): ")
        if check == "y" or check == "Y":
            print("Please wait while things initialize ...")
            log.log(
                "Started with following settings: Data: "
                + str(DATA)
                + " Train: "
                + str(TRAIN)
                + " SMOTE: "
                + str(SMOTE)
                + " Class Weight: "
                + str(CW)
                + ", Predict: "
                + str(PREDICT)
                + ", ROC: "
                + str(ROC)
                + ", BEST: "
                + str(BEST)
                + ", Models: "
                + str(MODEL_LIST),
                False,
            )
            main()
        else:
            print("Exiting ...")
            sys.exit()
    else:
        print("Please wait while things initialize ...")
        log.log(
            "Started with following settings: Data: "
            + str(DATA)
            + " Train: "
            + str(TRAIN)
            + " SMOTE: "
            + str(SMOTE)
            + " Class Weight: "
            + str(CW)
            + ", Predict: "
            + str(PREDICT)
            + ", ROC: "
            + str(ROC)
            + ", BEST: "
            + str(BEST)
            + ", Models: "
            + str(MODEL_LIST),
            False,
        )
        main()
