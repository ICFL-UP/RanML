
import sys
import log 
from log import Color
from optparse import OptionParser
import os
# Globals
FILE = ""
MODEL_LIST = ['RF','GBT', 'AB', 'KNN', 'DT', 'NB', 'LR', 'XGB', 'NN', 'SVM'] # 
DATA = 0
TRAIN = 0
PREDICT = 0
ROC = 0
BEST = 0
CM = 0


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
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    prefix = FILE[0:3]

    print(datetime.now())

    # # STATS 
    data = pd.read_csv(FILE)
    log.log(prefix + " Dataset stats - Category: " + str(data["category"].value_counts()))
    log.log(prefix + " Dataset stats - Label: \n" + str(data["label"].value_counts()))
    log.log(prefix + " Dataset stats - Shape: " + str(data.shape))


    if DATA:
        log.log("Preparing data splitting ...")
        data_reader.splitTrainTestVal(FILE) 


    log.log("Loading data ..")
    X = {
        "TRAIN": joblib.load("DATA/Train/"+prefix+"_features.pkl"),
        "VAL": joblib.load("DATA/Validation/"+prefix+"_features.pkl"),
        "TEST": joblib.load("DATA/Test/"+prefix+"_features.pkl")
    }
    Y = {
        "TRAIN": joblib.load("DATA/Train/"+prefix+"_labels.pkl"),
        "VAL": joblib.load("DATA/Validation/"+prefix+"_labels.pkl"),
        "TEST": joblib.load("DATA/Test/"+prefix+"_labels.pkl")
    }

    
    log.log("Dataset TRAIN Y: \n" + str(Y["TRAIN"].value_counts()))
    log.log("Dataset VAL Y: \n" + str(Y["VAL"].value_counts()))
    log.log("Dataset TEST Y: \n" + str(Y["TEST"].value_counts()))


    if TRAIN:
        # Classifier Training
        log.log("\n\nTraining Classifiers ...")
        if "RF" in MODEL_LIST:
            try:
                classifiers.randomForrest(X["TRAIN"], Y["TRAIN"], prefix)
            except:
                print(traceback.print_exc())
                log.log("\n\n\n\n\nERROR in training of RF\n\n\n\n")
        
        if "GBT" in MODEL_LIST:
            try:
                classifiers.gradientBoost(X["TRAIN"], Y["TRAIN"], prefix)
            except:
                print(traceback.print_exc())   
                log.log("\n\n\n\n\nERROR in training of DT\n\n\n\n")

        if "AB" in MODEL_LIST:
            try:
                classifiers.adaBoost(X["TRAIN"], Y["TRAIN"], prefix)
            except:
                print(traceback.print_exc())
                log.log("\n\n\n\n\nERROR in training of AB\n\n\n\n")
        
        if "KNN" in MODEL_LIST:
            try:
                classifiers.knn(X["TRAIN"], Y["TRAIN"], prefix)
            except:
                print(traceback.print_exc())
                log.log("\n\n\n\n\nERROR in training of KNN\n\n\n\n")

        if "DT" in MODEL_LIST:
            try:
                classifiers.decisionTree(X["TRAIN"], Y["TRAIN"], prefix)
            except:
                print(traceback.print_exc())   
                log.log("\n\n\n\n\nERROR in training of DT\n\n\n\n")

        if "NB" in MODEL_LIST:
            try:
                le = LabelEncoder()
                y_train = le.fit_transform(Y["TRAIN"])
                classifiers.naiveBayes(X["TRAIN"], y_train, prefix)
            except:
                print(traceback.print_exc())   
                log.log("\n\n\n\n\nERROR in training of NB\n\n\n\n")

        if "XGB" in MODEL_LIST:
            try:
                le = LabelEncoder()
                y_train = le.fit_transform(Y["TRAIN"])
                classifiers.xgboost(X["TRAIN"], y_train, prefix)
            except:
                print(traceback.print_exc())   
                log.log("\n\n\n\n\nERROR in training of XGB\n\n\n\n")
        
        if "LR" in MODEL_LIST:
            try:
                le = LabelEncoder()
                y_train = le.fit_transform(Y["TRAIN"])
                classifiers.logisticRegression(X["TRAIN"], y_train, prefix)
            except:
                print(traceback.print_exc())
                log.log("\n\n\n\n\nERROR in training of LR\n\n\n\n")
        
        if "KM" in MODEL_LIST:
            try:
                classifiers.kmeans(X["TRAIN"], Y["TRAIN"], prefix)
            except:
                print(traceback.print_exc())
                log.log("\n\n\n\n\nERROR in training of KM\n\n\n\n")
        
        if "NN" in MODEL_LIST:
            try:
                classifiers.nn(X["TRAIN"], Y["TRAIN"], prefix)
            except:
                print(traceback.print_exc())
                log.log("\n\n\n\n\nERROR in training of NN\n\n\n\n")

        if "SVM" in MODEL_LIST:
            try:
                classifiers.svm(X["TRAIN"], Y["TRAIN"], prefix)
            except:
                print(traceback.print_exc())
                log.log("\n\n\n\n\nERROR in training of SVM\n\n\n\n")
        

    
    if PREDICT:

        # Predict
        log.log("\n\nPREDICTING ...\n\n")
        results = []
        models = {}
        le = LabelEncoder()
        for mdl in MODEL_LIST:
            models[mdl+"_"+prefix] = joblib.load('Models/{}_{}_model.pkl'.format(mdl, prefix))
            if mdl in ['LR', 'NB', 'XGB', 'KM']:
                y_val = [ 1 if x == 'M' else 0 for x in Y['VAL']]
                results.append(classifiers.evaluate_model(mdl+"_"+prefix, models[mdl+"_"+prefix], X["VAL"], y_val, prefix))
            else:
                results.append(classifiers.evaluate_model(mdl+"_"+prefix, models[mdl+"_"+prefix], X["VAL"], Y["VAL"], prefix))
        from tabulate import tabulate
        log.log("\n\n"+tabulate(results, headers=["Name", "TP", "TN", "FP", "FN", "Recall", "Precision", "F1-Score", "AUC", "LogLoss", "Latency(ms)", "Num", "Accuracy"])+"\n\n\n")


    if ROC:        
        ##ROC Curve
        print("\n\n\nGenerating ROC\n\n")
        models = {}
        for mdl in MODEL_LIST:
            models[mdl+"_"+prefix] = joblib.load('Models/{}_{}_model.pkl'.format(mdl, prefix))
        fig = plt.figure(figsize=(7, 7), dpi=300)
        axes = fig.gca()
        for x in MODEL_LIST:
            if x in ['LR', 'NB', 'XGB', 'KM']:
                y_test = [ 1 if x == 'M' else 0 for x in Y['VAL']]
                RocCurveDisplay.from_estimator(models[x+"_"+prefix], X["VAL"], y_test, ax=axes)
            else:
                RocCurveDisplay.from_estimator(models[x+"_"+prefix], X["VAL"], Y["VAL"], ax=axes)
                
        plt.savefig("ROC.png")

    if CM:
        # Confusion Matrix
        print("\n\n\nGenerating Confusion Matrix\n\n\n")
        models = {}
        for mdl in MODEL_LIST:
            models[mdl+"_"+prefix] = joblib.load('Models/{}_{}_model.pkl'.format(mdl, prefix))

        if not os.path.exists("CM"):
            os.mkdir("CM")
        for x in MODEL_LIST:
            fig = plt.figure(figsize=(7, 7), dpi=300)
            axes = fig.gca()
            if x in ['LR', 'NB', 'XGB', 'KM']:
                y_test = [ 1 if x == 'M' else 0 for x in Y['VAL']]
                ConfusionMatrixDisplay.from_estimator(models[x+"_"+prefix], X["VAL"], y_test, ax=axes)
            else:
                ConfusionMatrixDisplay.from_estimator(models[x+"_"+prefix], X["VAL"], Y["VAL"], ax=axes)
                
            plt.savefig("CM/"+x+"_"+prefix+".png")

    if BEST:
        # # BEST MODELS
        print("\n\n\nBEST MODELS\n\n")
        mdl = "RF"
        log.log("Training bagging on " + mdl)
        best = joblib.load('Models/{}_{}_model.pkl'.format(mdl, prefix))
        classifiers.bagging_best(X['TRAIN'], Y['TRAIN'], prefix, best)

        results = classifiers.evaluate_model("BEST_"+prefix, best, X["TEST"], Y["TEST"], prefix)
        from tabulate import tabulate
        log.log(tabulate(results, headers=["Name", "TP", "TN", "FP", "FN", "Recall", "Precision", "F1-Score", "AUC", "LogLoss", "Latency(ms)", "Num", "Accuracy"]))

        



if __name__ == "__main__":
    print("\n\nWelcome to "+Color.GREEN+"Ran"+Color.BLUE+"For"+Color.RED+"Red\n\n"+Color.END)
    if len(sys.argv[1:]) == 0:
        print("No arguments passed, please use -h or --help for help")
        sys.exit()


    parser = OptionParser(option_class=log.MultipleOption)
    parser.add_option("-d", "--data", action="store_true", help="Preprocess the data from the CSV file")
    parser.add_option("-t", "--train", action="store_true", help="Train the models")
    parser.add_option("-p", "--predict", action="store_true", help="Predict from the trained models")
    parser.add_option("-r", "--roc", action="store_true", help="Produce the ROC Curve of the models predictions")
    parser.add_option("-b", "--best", action="store_true",help="Run the best models and compare on new data")
    parser.add_option("-c", "--confusion", action="store_true",help="Produce the Confusion matrix image for the models")
    parser.add_option("-i", "--input", dest="input", help="Specify the input file")
    parser.add_option("-s", "--silent", action="store_true", help="Silent - no prompts to verify")
    parser.add_option("-m", "--models", dest="models", metavar='MODELS', action="extend", 
                    help="Select the Models to be applied, by default all are applied, multiple can be selected comma-delimited. Valid choices are: %s." % (MODEL_LIST))

    (options, args) = parser.parse_args()
    
    if options.data != None:
        DATA = 1
    if options.train != None:
        TRAIN = 1
    if options.predict != None:
        PREDICT = 1
    if options.roc != None:
        ROC = 1    
    if options.confusion != None:
        CM = 1
    if options.best != None:
        BEST = 1
    if options.input == None:
        print("-i flag for the Input file must be specified")
        sys.exit()
    else:
        if not options.input.endswith(".csv"):
            print("Please use a csv file for training")
            sys.exit()
        if not os.path.isfile(options.input):
            print('Error: File does not exist')
            sys.exit()
        FILE = options.input   
    if options.models != None:
        if options.train != None or options.predict != None or options.roc != None:
            newList = []
            for m in options.models:
                if m in MODEL_LIST:
                    newList.append(m)
            MODEL_LIST = newList
        else:
            print("-m can only be used with the following flags [-t, -p, -r]")
    if options.train == None and options.data == None and options.predict == None and options.roc == None and options.best == None and options.confusion == None:
        print("No valid arguments passed, please use -h or --help for help")
        sys.exit()

    
    print("Your choices: ")
    print("Data: ", Color.GREEN + "True" + Color.END if DATA == 1 else "False")
    print("Train: ", Color.GREEN + "True" + Color.END if TRAIN == 1 else "False")
    print("Predict: ", Color.GREEN + "True" + Color.END if PREDICT == 1 else "False")
    print("ROC: ", Color.GREEN + "True" + Color.END if ROC == 1 else "False")
    print("CM: ", Color.GREEN + "True" + Color.END if CM == 1 else "False")
    print("BEST: ", Color.GREEN + "True" + Color.END if BEST == 1 else "False")
    print("Models: ", MODEL_LIST)

    if not options.silent:
        check = input("\nPlease verify your choices (Y/N): ")
        if check=="y" or check=="Y":
            print("Please wait while things initialize ...")
            log.log("Started with following settings: Data: " + str(DATA) + " Train: " + str(TRAIN) + ", Predict: " + str(PREDICT) + ", ROC: " + str(ROC) + ", BEST: " + str(BEST) + ", Models: " + str(MODEL_LIST), False)
            main()
        else:
            print("Exiting ...")
            sys.exit()
    else:
        print("Please wait while things initialize ...")
        log.log("Started with following settings: Data: " + str(DATA) + " Train: " + str(TRAIN) + ", Predict: " + str(PREDICT) + ", ROC: " + str(ROC) + ", BEST: " + str(BEST) + ", Models: " + str(MODEL_LIST), False)
        main()
