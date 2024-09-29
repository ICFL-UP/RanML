import log
import os
import time
import numpy as np
from sklearn.svm import SVC
from joblib import parallel_backend
import joblib
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

CV = 10

### 1 = Ransomware; 0 = Benign

def print_results(results, classifier):
    means = results.cv_results_["mean_test_score"]
    stds = results.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, results.cv_results_["params"]):
        log.log("{} (+/-{}) for {}".
                format(round(mean, 4), round(std * 2, 4), params))
    log.log("BEST PARAMS for {}: {}\n".
            format(classifier, results.best_params_))


def evaluate_model(name, model, features, labels, prefix):
    start = time.time()
    pred = model.predict(features)
    end = time.time()
    accuracy = round(accuracy_score(labels, pred), 4)*100
    auc = "-"
    logloss = "-"
    if name in ["XGB_" + prefix, "NB_" + prefix, "LR_" + prefix]:
        precision = round(precision_score(labels, pred, pos_label=1), 4)
        recall = round(recall_score(labels, pred, pos_label=1), 4)
        f1 = round(f1_score(labels, pred, pos_label=1), 4)
    else:
        if name in ["KM"]:
            precision = round(
                precision_score(
                    labels, pred, pos_label=1, average="micro"), 4
            )
            recall = round(
                recall_score(labels, pred, pos_label=1, average="micro"), 4
            )
            f1 = round(
                f1_score(labels, pred, pos_label=1, average="micro"), 4
            )
            
        else:
            precision = round(precision_score(labels, pred, pos_label=1), 4)
            recall = round(recall_score(labels, pred, pos_label=1), 4)
            f1 = round(f1_score(labels, pred, pos_label=1), 4)
    
            auc = round(roc_auc_score(labels, model.predict_proba(features)[:, 1]), 4)
            logloss = round(log_loss(labels, model.predict_proba(features)), 4)

    cm = confusion_matrix(labels, pred)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    lat = round((end - start) * 1000, 4)
    log.log(
        log.Color.GREEN
        + "\n\n{} -- ".format(name)
        + log.Color.END
        + "\nTP: {} \nTN: {} \nFP: {} \nFN: {} \nRecall: {} \nPrecision: {} \
            \nF1-Score: {}  \nAUC: {} \nLogLoss: {} \nLatency: {} ms \
            \nAccuracy: {}\n\n".format(
            TP[0],
            TN[0],
            FP[0],
            FN[0],
            recall,
            precision,
            f1,
            auc,
            logloss,
            lat,
            accuracy,
        )
    )
    # log.log("{} & {} & {} & {} & {} & {} & {} \n{} & {} & {} & {} & {}".
    # format(TP[0], TN[0], FP[0], FN[0], recall, precision, f1, accuracy,
    # auc, logloss, lat, round(lat/len(pred),4)))
    return [
        name,
        TP[0],
        TN[0],
        FP[0],
        FN[0],
        recall,
        precision,
        f1,
        auc,
        logloss,
        lat,
        accuracy,
    ]


# ====================================================================
# ================  Classifiers
# ====================================================================
def randomForrest(train_data, correct_class, nlp, cw=None):
    log.log("\nTraining RandomForrest for {}...".format(nlp))
    with parallel_backend("threading", n_jobs=os.cpu_count()):

        start_time = time.time()
        if cw:
            rf = RandomForestClassifier(class_weight=cw)
        else:
            rf = RandomForestClassifier(class_weight=cw)
        param = {
            "n_estimators": [5, 50, 250],
            "max_depth": [2, 4, 8, 16, 32, None],
            "criterion": ["gini", "entropy", "log_loss"],
        }

        cv = GridSearchCV(rf, param, cv=CV, verbose=10)
        cv.fit(train_data, correct_class.to_numpy())
        print_results(cv, "RandomForrest + {}".format(nlp))
        joblib.dump(cv.best_estimator_, "Models/RF_{}_model.pkl".format(nlp))
        log.log(
            "Train time for Random Forrest + {}: ".format(nlp)
            + str((time.time() - start_time) / 60)
            + " min"
        )
        return cv.best_estimator_


def gradientBoost(train_data, correct_class, nlp):
    log.log("\nTraining Gradient Boost for {}...".format(nlp))
    with parallel_backend("threading", n_jobs=os.cpu_count()):

        start_time = time.time()
        gbt = GradientBoostingClassifier()
        param = {
            "n_estimators": [5, 50, 250],
            "max_depth": [2, 4, 8, 16, 32, None],
            "loss": ["log_loss", "exponential"],
            "learning_rate": [0.01, 0.1, 1, 10, 100],
        }

        cv = GridSearchCV(gbt, param, cv=CV, verbose=10)
        cv.fit(train_data, correct_class.to_numpy())
        print_results(cv, "Gradient Boosted Tree + {}".format(nlp))
        joblib.dump(cv.best_estimator_, "Models/GBT_{}_model.pkl".format(nlp))
        log.log(
            "Train time for Gradient Boosted Tree + {}: ".format(nlp)
            + str((time.time() - start_time) / 60)
            + " min"
        )
        return cv.best_estimator_


def adaBoost(train_data, correct_class, nlp):
    log.log("\nTraining AdaBoost for {}...".format(nlp))
    with parallel_backend("threading", n_jobs=os.cpu_count()):
        start_time = time.time()
        classifier = AdaBoostClassifier()
        param = {
            "n_estimators": [5, 50, 250],
            "learning_rate": [0.01, 0.1, 1, 10, 100]
        }

        cv = GridSearchCV(classifier, param, cv=CV, verbose=10)
        cv.fit(train_data, correct_class.to_numpy())
        print_results(cv, "AdaBoost + {}".format(nlp))
        joblib.dump(cv.best_estimator_, "Models/AB_{}_model.pkl".format(nlp))
        log.log(
            "Train time for AdaBoost + {}: ".format(nlp)
            + str((time.time() - start_time) / 60)
            + " min"
        )
        return cv.best_estimator_


def bagging(train_data, correct_class, nlp):
    log.log("\nTraining Bagging for {}...".format(nlp))
    with parallel_backend("threading", n_jobs=os.cpu_count()):
        start_time = time.time()
        classifier = BaggingClassifier()
        param = {
            "n_estimators": [5, 10, 50, 250],
            "estimator": [
                RandomForestClassifier(),
                DecisionTreeClassifier(),
            ],
        }

        cv = GridSearchCV(classifier, param, cv=CV, verbose=10)
        cv.fit(train_data, correct_class.to_numpy())
        print_results(cv, "Bagging + {}".format(nlp))
        joblib.dump(cv.best_estimator_, "Models/BAG_{}_model.pkl".format(nlp))
        log.log(
            "Train time for Bagging + {}: ".format(nlp)
            + str((time.time() - start_time) / 60)
            + " min"
        )
        return cv.best_estimator_


def knn(train_data, correct_class, nlp):
    log.log("\nTraining KNN for {}...".format(nlp))
    with parallel_backend("threading", n_jobs=os.cpu_count()):
        start_time = time.time()
        classifier = KNeighborsClassifier()
        param = {
            "n_neighbors": [5, 50, 250],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
            "leaf_size": [5, 50, 250],
        }

        cv = GridSearchCV(classifier, param, cv=CV, verbose=10)
        cv.fit(train_data, correct_class.to_numpy())
        print_results(cv, "KNN + {}".format(nlp))
        joblib.dump(cv.best_estimator_, "Models/KNN_{}_model.pkl".format(nlp))
        log.log(
            "Train time for KNN + {}: ".format(nlp)
            + str((time.time() - start_time) / 60)
            + " min"
        )
        return cv.best_estimator_


def decisionTree(train_data, correct_class, nlp, cw=None):
    log.log("\nTraining Decision Tree for {}...".format(nlp))
    with parallel_backend("threading", n_jobs=os.cpu_count()):
        start_time = time.time()
        if cw:
            classifier = DecisionTreeClassifier(class_weight=cw)
        else:
            classifier = DecisionTreeClassifier()
        param = {
            "criterion": ["gini", "entropy", "log_loss"],
            "splitter": ["best", "random"],
        }

        cv = GridSearchCV(classifier, param, cv=CV, verbose=10)
        cv.fit(train_data, correct_class.to_numpy())
        print_results(cv, "DT + {}".format(nlp))
        joblib.dump(cv.best_estimator_, "Models/DT_{}_model.pkl".format(nlp))
        log.log(
            "Train time for DT + {}: ".format(nlp)
            + str((time.time() - start_time) / 60)
            + " min"
        )
        return cv.best_estimator_


def naiveBayes(train_data, correct_class, nlp):
    log.log("\nTraining Naive Bayes for {}...".format(nlp))
    with parallel_backend("threading", n_jobs=os.cpu_count()):
        start_time = time.time()
        classifier = GaussianNB()
        param = {"var_smoothing": np.logspace(0, -9, num=100)}

        cv = GridSearchCV(classifier, param, cv=CV, verbose=10)
        cv.fit(train_data, correct_class.to_numpy())
        print_results(cv, "NB + {}".format(nlp))
        joblib.dump(cv.best_estimator_, "Models/NB_{}_model.pkl".format(nlp))
        log.log(
            "Train time for NB + {}: ".format(nlp)
            + str((time.time() - start_time) / 60)
            + " min"
        )
        return cv.best_estimator_


def xgboost(train_data, correct_class, nlp):
    log.log("\nTraining XGB for {}...".format(nlp))
    with parallel_backend("threading", n_jobs=os.cpu_count()):
        start_time = time.time()
        classifier = XGBClassifier()
        param = {
            "booster": ["gbtree", "gblinear", "dart"],
            "learning_rate": [0.01, 0.1, 1, 10, 100],
        }

        cv = GridSearchCV(classifier, param, cv=CV, verbose=10)
        cv.fit(train_data, correct_class.to_numpy())
        print_results(cv, "XGB + {}".format(nlp))
        joblib.dump(cv.best_estimator_, "Models/XGB_{}_model.pkl".format(nlp))
        log.log(
            "Train time for XGB + {}: ".format(nlp)
            + str((time.time() - start_time) / 60)
            + " min"
        )
        return cv.best_estimator_


def logisticRegression(train_data, correct_class, nlp, cw=None):
    log.log("\nTraining LR for {}...".format(nlp))
    with parallel_backend("threading", n_jobs=os.cpu_count()):
        start_time = time.time()
        if cw:
            classifier = LogisticRegression(class_weight=cw)
        else:
            classifier = LogisticRegression()
        param = {
            "penalty": ["l2"],
            "C": [100, 10, 1.0, 0.1, 0.01],
            "solver": ['lbfgs'],
#            "max_iter": [100, 1000, 2500, 5000],
        }

        cv = GridSearchCV(classifier, param, cv=CV, verbose=10)
        cv.fit(
            StandardScaler().fit_transform(train_data),  # Scale the data
            correct_class.to_numpy())
        print_results(cv, "LR + {}".format(nlp))
        joblib.dump(cv.best_estimator_, "Models/LR_{}_model.pkl".format(nlp))
        log.log(
            "Train time for LR + {}: ".format(nlp)
            + str((time.time() - start_time) / 60)
            + " min"
        )
        return cv.best_estimator_


def kmeans(train_data, correct_class, nlp):
    log.log("\nTraining KM for {}...".format(nlp))
    with parallel_backend("threading", n_jobs=os.cpu_count()):
        start_time = time.time()
        classifier = KMeans()

        param = {
            "n_clusters": [2],
            "init": ["k-means++", "random"],
            "n_init": [5, 10, 15],
            "max_iter": [100, 200, 300, 400, 500],
            "tol": [0.0001, 0.001, 0.01],
            "algorithm": ["llyod", "full", "elkan"],
            "random_state": [0, 42, 100],
        }

        cv = GridSearchCV(classifier, param, cv=CV, verbose=10)
        cv.fit(train_data, correct_class.to_numpy())
        print_results(cv, "KM + {}".format(nlp))
        joblib.dump(cv.best_estimator_, "Models/KM_{}_model.pkl".format(nlp))
        log.log(
            "Train time for KM + {}: ".format(nlp)
            + str((time.time() - start_time) / 60)
            + " min"
        )
        return cv.best_estimator_


def nn(train_data, correct_class, nlp):
    log.log("\nTraining NN for {}...".format(nlp))
    with parallel_backend("threading", n_jobs=os.cpu_count()):
        start_time = time.time()
        classifier = MLPClassifier()
        param = {
            "hidden_layer_sizes": [(10, 30, 10), (20,)],
            "activation": ["tanh", "relu"],
            "solver": ["sgd", "adam"],
            "alpha": [0.0001, 0.05],
            "learning_rate": ["constant", "adaptive"],
        }

        cv = GridSearchCV(classifier, param, cv=CV, verbose=10)
        cv.fit(train_data, correct_class.to_numpy())
        print_results(cv, "NN + {}".format(nlp))
        joblib.dump(cv.best_estimator_, "Models/NN_{}_model.pkl".format(nlp))
        log.log(
            "Train time for NN + {}: ".format(nlp)
            + str((time.time() - start_time) / 60)
            + " min"
        )
        return cv.best_estimator_


def svm(train_data, correct_class, nlp, cw=None):
    log.log("\nTraining SVM for {}...".format(nlp))
    with parallel_backend("threading", n_jobs=os.cpu_count()):
        start_time = time.time()
        if cw:
            classifier = SVC(verbose=True, class_weight=cw)
        else:
            classifier = SVC(verbose=True)
        param = {
            "kernel": ["rbg", "sigmoid"],
            "C": [0.1, 1, 10],
            "probability": [True]
        }

        cv = GridSearchCV(classifier, param, cv=CV, verbose=10)
        cv.fit(train_data, correct_class.to_numpy())
        print_results(cv, "SVM + {}".format(nlp))
        joblib.dump(cv.best_estimator_, "Models/SVM_{}_model.pkl".format(nlp))
        log.log(
            "Train time for SVM + {}: ".format(nlp)
            + str((time.time() - start_time) / 60)
            + " min"
        )
        return cv.best_estimator_


def bagging_best(train_data, correct_class, nlp, estimator):
    log.log("\nTraining Bagging for {}...".format(nlp))
    with parallel_backend("threading", n_jobs=os.cpu_count()):
        start_time = time.time()
        classifier = BaggingClassifier(estimator=estimator)
        param = {
            "n_estimators": [5, 10, 50, 250],
        }

        cv = GridSearchCV(classifier, param, cv=CV, verbose=10)
        cv.fit(train_data, correct_class.to_numpy())
        print_results(cv, "Bagging + {}".format(nlp))
        joblib.dump(cv.best_estimator_, "Models/BEST_{}_model.pkl".format(nlp))
        log.log(
            "Train time for Bagging + {}: ".format(nlp)
            + str((time.time() - start_time) / 60)
            + " min"
        )
        return cv.best_estimator_
