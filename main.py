
import sys
import log 
from log import Color
from optparse import OptionParser

# Globals
MODEL_LIST = ['RF','GBT', 'AB', 'KNN', 'DT', 'NB', 'XGB', 'LR', 'KM', 'NN', 'SVM']
DATA = 0
TRAIN = 0
PREDICT = 0
ROC = 0
BEST = 0

print("Welcome to RanForRed")
if len(sys.argv[1:]) == 0:
    print("No arguments passed, please use -h or --help for help")
    sys.exit()

if __name__ == "__main__":
    parser = OptionParser(option_class=log.MultipleOption)
    parser.add_option("-d", "--data", action="store_true", help="Preprocess the data from the CSV file")
    parser.add_option("-t", "--train", action="store_true", help="Train the models")
    parser.add_option("-p", "--predict", action="store_true", help="Predict from the trained models")
    parser.add_option("-r", "--roc", action="store_true", help="Produce the ROC Curve of the models predictions")
    parser.add_option("-b", "--best", action="store_true",help="Run the best models and compare on new data")
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
    if options.best != None:
        BEST = 1

    if options.models != None:
        if options.train != None or options.predict != None or options.roc != None:
            newList = []
            for m in options.models:
                if m in MODEL_LIST:
                    newList.append(m)
            MODEL_LIST = newList
        else:
            print("-m can only be used with the following flags [-t, -p, -r]")
    if options.train == None and options.data == None and options.predict == None and options.roc == None and options.best == None:
        print("No valid arguments passed, please use -h or --help for help")
        sys.exit()

    print("Your choices: ")
    print("Data: ", Color.GREEN + "True" + Color.END if DATA == 1 else "False")
    print("Train: ", Color.GREEN + "True" + Color.END if TRAIN == 1 else "False")
    print("Predict: ", Color.GREEN + "True" + Color.END if PREDICT == 1 else "False")
    print("ROC: ", Color.GREEN + "True" + Color.END if ROC == 1 else "False")
    print("BEST: ", Color.GREEN + "True" + Color.END if BEST == 1 else "False")
    print("Models: ", MODEL_LIST)

    check = input("\nPlease verify your choices (Y/N): ")
    if check=="y" or check=="Y":
        print("Please wait while things init.")
        # main()
    else:
        print("Exiting ...")
        sys.exit()


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

    data_filename = "5_Entropy_of_PE_Sections_versioned.csv"
    prefix = data_filename[0:3]

    print(datetime.now())

    # # STATS 
    data = pd.read_csv(data_filename)
    log.log("Dataset stats - Category: " + str(data["category"].value_counts()))
    log.log("Dataset stats - Label: \n" + str(data["label"].value_counts()))
    log.log("Dataset stats - Shape: " + str(data.shape))


    if DATA:
        log.log("Preparing data splitting ...")
        data_reader.splitTrainTestVal(data_filename) 


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
        try:
            classifiers.randomForrest(X["TRAIN"], Y["TRAIN"], "RF")
        except:
            print(traceback.print_exc())
            log.log("\n\n\n\n\nERROR in training of RF\n\n\n\n")
        
        try:
            classifiers.gradientBoost(X["TRAIN"], Y["TRAIN"], "GBT")
        except:
            print(traceback.print_exc())   
            log.log("\n\n\n\n\nERROR in training of DT\n\n\n\n")

        try:
            classifiers.adaBoost(X["TRAIN"], Y["TRAIN"], "AB")
        except:
            print(traceback.print_exc())
            log.log("\n\n\n\n\nERROR in training of AB\n\n\n\n")
            
        try:
            classifiers.knn(X["TRAIN"], Y["TRAIN"], "KNN")
        except:
            print(traceback.print_exc())
            log.log("\n\n\n\n\nERROR in training of KNN\n\n\n\n")

        try:
            classifiers.decisionTree(X["TRAIN"], Y["TRAIN"], "DT")
        except:
            print(traceback.print_exc())   
            log.log("\n\n\n\n\nERROR in training of DT\n\n\n\n")

        try:
            le = LabelEncoder()
            y_train = le.fit_transform(Y["TRAIN"])
            classifiers.naiveBayes(X["TRAIN"], y_train, "NB")
        except:
            print(traceback.print_exc())   
            log.log("\n\n\n\n\nERROR in training of NB\n\n\n\n")

        try:
            le = LabelEncoder()
            y_train = le.fit_transform(Y["TRAIN"])
            classifiers.xgboost(X["TRAIN"], y_train, "XGB")
        except:
            print(traceback.print_exc())   
            log.log("\n\n\n\n\nERROR in training of XGB\n\n\n\n")
        
        try:
            classifiers.logisticRegression(X["TRAIN"], Y["TRAIN"], "LR")
        except:
            print(traceback.print_exc())
            log.log("\n\n\n\n\nERROR in training of LR\n\n\n\n")
        
        try:
            classifiers.kmeans(X["TRAIN"], Y["TRAIN"], "KM")
        except:
            print(traceback.print_exc())
            log.log("\n\n\n\n\nERROR in training of KM\n\n\n\n")
        
        try:
            classifiers.nn(X["TRAIN"], Y["TRAIN"], "NN")
        except:
            print(traceback.print_exc())
            log.log("\n\n\n\n\nERROR in training of NN\n\n\n\n")

        try:
            classifiers.svm(X["TRAIN"], Y["TRAIN"], "SVM")
        except:
            print(traceback.print_exc())
            log.log("\n\n\n\n\nERROR in training of SVM\n\n\n\n")
        

    
    if PREDICT:

        # Predict
        log.log("\n\nPREDICTING ...\n\n")
        models = {}
        le = LabelEncoder()
        for mdl in MODEL_LIST:
            models[mdl+"_"+mdl] = joblib.load('Models/{}_{}_model.pkl'.format(mdl, mdl))
            if mdl in ['NB', 'XGB']:
                y_val = [ 1 if x == 'M' else 0 for x in Y['VAL']]
                classifiers.evaluate_model(mdl+"_"+mdl, models[mdl+"_"+mdl], X["VAL"], y_val)
            elif mdl == "SVM":
                y_val = [ 1 if x == 'M' else 0 for x in Y['VAL']]
                classifiers.evaluate_model(mdl+"_"+mdl, models[mdl+"_"+mdl], X["VAL"], y_val)
            else:
                classifiers.evaluate_model(mdl+"_"+mdl, models[mdl+"_"+mdl], X["VAL"], Y["VAL"])

    if ROC:        
        ##ROC Curve
        print("\n\n\nGenerating ROC\n\n")
        models = {}
        for mdl in MODEL_LIST:
            models[mdl+"_"+mdl] = joblib.load('Models/{}_{}_model.pkl'.format(mdl, mdl))
        fig = plt.figure(figsize=(7, 7), dpi=300)
        axes = fig.gca()
        for x in MODEL_LIST:
            if x in ["NB", "XGB"]:
                y_test = [ 1 if x == 'M' else 0 for x in Y['VAL']]
                RocCurveDisplay.from_estimator(models[x+"_"+x], X["VAL"], y_test, ax=axes)
            else:
                RocCurveDisplay.from_estimator(models[x+"_"+x], X["VAL"], Y["VAL"], ax=axes)
        plt.savefig("ROC.png")

    if BEST:
        # # BEST MODELS
        print("\n\n\nBEST MODELS\n\n")
        

