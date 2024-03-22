import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import seed
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import confusion_matrix
from collections import Counter
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import csv
from sklearn.metrics import precision_recall_curve

seed(1)


###
def XGB_cases(file, OUT, IN):
    XY = pd.read_csv('XY_' + file + '.csv')

    print(XY[XY['Label'] == 0]['Label'].count())
    print(XY[XY['Label'] == 1]['Label'].count())

    XY = XY.drop(columns=['ALT (GPT) [S]', 'AST (GOT) [S]', 'PLATELETS', 'ALT (GPT) [S]_delta', 'AST (GOT) [S]_delta',
                          'PLATELETS_delta'])

    y = XY['Label'].values
    XY = XY.drop(columns=['Label'])

    # NaN analisys and predictors selection
    nan_percentage = XY.isna().sum() / len(XY) * 100
    th_nan = nan_percentage.values < 90
    XY = XY.loc[:, th_nan]
    #

    # Predictor names: sorting alphabetically
    predictors = XY.columns.tolist()
    predictors.sort()
    XY = XY[predictors]
    #

    # Extra-value data imputation
    XY = XY.fillna(-999)
    X = XY.values
    #

    ###

    outer_f1 = []
    outer_precision = []
    outer_recall = []
    outer_accuracy = []
    outer_auc = []
    outer_pr_auc = []

    nrows = 2
    ncols = 2
    cm_tot = np.zeros((nrows, ncols))

    nrowss = 1
    ncolss = len(X[0])
    importance_tot = np.zeros((nrowss, ncolss))

    # Cross-Validation
    skf_ext = StratifiedKFold(n_splits=OUT, shuffle=True, random_state=1)
    skf_int = StratifiedKFold(n_splits=IN, shuffle=True, random_state=1)
    skf_ext.get_n_splits(X, y)
    fold = 0

    for train_index, test_index in skf_ext.split(X, y):
        fold = fold + 1
        print('Fold:')
        print(fold)
        X_train_outer, X_test_outer = X[train_index], X[test_index]
        y_train_outer, y_test_outer = y[train_index], y[test_index]

        # SMOTE
        counter = Counter(y_train_outer.ravel())
        print(counter)
        oversample = SMOTE()
        X_train_outer, y_train_outer = oversample.fit_resample(X_train_outer, y_train_outer)
        counter = Counter(y_train_outer)
        print(counter)
        #

        classifier = xgb.XGBClassifier(objective='binary:logistic', random_state=1, missing=-999,
                                       importance_type='weight')

        parameters = {'n_estimators': [150, 200],
                      'max_depth': [25, 50, 75]}

        model = GridSearchCV(estimator=classifier, param_grid=[parameters], cv=skf_int, scoring='roc_auc')
        model.fit(X_train_outer, y_train_outer)

        best_pars = model.best_params_
        print(best_pars)

        best_model = model.best_estimator_

        best_model.fit(X_train_outer, y_train_outer)
        outer_posteriors = best_model.predict_proba(X_test_outer)
        y_pred = best_model.predict(X_test_outer)

        outer_auc.append(roc_auc_score(y_test_outer, outer_posteriors[:, 1]))
        precision_aux, recall_aux, _ = precision_recall_curve(y_test_outer, outer_posteriors[:, 1], pos_label=1)
        outer_pr_auc.append(auc(recall_aux, precision_aux))
        outer_f1.append(f1_score(y_test_outer, y_pred, average='macro'))
        outer_recall.append(recall_score(y_test_outer, y_pred, average='macro'))
        outer_precision.append(precision_score(y_test_outer, y_pred, average='macro'))
        outer_accuracy.append(accuracy_score(y_test_outer, y_pred))

        cm = confusion_matrix(y_test_outer, y_pred)
        cm_tot = cm_tot + cm

        importance_tot = importance_tot + best_model.feature_importances_

    print('Accuracy: %.4f' % (np.mean(outer_accuracy)))
    print('F1: %.4f' % (np.mean(outer_f1)))
    print('Precision: %.4f' % (np.mean(outer_precision)))
    print('Recall: %.4f' % (np.mean(outer_recall)))
    print('AUC: %.4f' % (np.mean(outer_auc)))
    print('PR_AUC: %.4f' % (np.mean(outer_pr_auc)))
    print(cm_tot)

    cm_tot = cm_tot.astype(int)
    specificity = cm_tot[0, 0] / (cm_tot[0, 0] + cm_tot[0, 1])
    sensitivity = cm_tot[1, 1] / (cm_tot[1, 0] + cm_tot[1, 1])

    cm_tot1 = np.array([[cm_tot[0, 0] / (cm_tot[0, 0] + cm_tot[0, 1]), cm_tot[0, 1] / (cm_tot[0, 0] + cm_tot[0, 1])],
                        [cm_tot[1, 0] / (cm_tot[1, 0] + cm_tot[1, 1]), cm_tot[1, 1] / (cm_tot[1, 0] + cm_tot[1, 1])]])

    # Plot confusion matrix
    ax = plt.subplot()
    sns.heatmap(cm_tot, annot=True, annot_kws={'va': 'bottom'}, fmt='d', ax=ax)
    sns.heatmap(cm_tot1, annot=True, annot_kws={'va': 'top'}, fmt=".1%", cbar=False)
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['control', 'fibrosis'])
    ax.yaxis.set_ticklabels(['control', 'fibrosis'])
    ##

    # Plot features importance
    indices = np.argsort(importance_tot)[::-1]
    indices = indices.ravel()
    indices = indices[-10:]
    names = [predictors[i] for i in indices]
    importance_tot = np.transpose(importance_tot)
    importance_tot = importance_tot.ravel()
    plt.figure()
    plt.barh(range(indices.shape[0]), importance_tot[indices])
    plt.yticks(range(indices.shape[0]), names)
    plt.xlabel('Features Importance')
    ##

    # Save predictive performance results to csv
    with open('XGB_results_' + file + '.csv', 'w') as csvfile:
        fieldnames = ['Accuracy', 'Accuracystd', 'F1', 'F1std', 'Precision', 'Precisionstd', 'Recall', 'Recallstd',
                      'AUC',
                      'AUCstd', 'PRAUC', 'PRAUCstd', 'Specificity', 'Sensitivity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow(
            {'Accuracy': np.mean(outer_accuracy, dtype=np.float16),
             'Accuracystd': np.std(outer_accuracy, dtype=np.float16),
             'F1': np.mean(outer_f1, dtype=np.float16), 'F1std': np.std(outer_f1, dtype=np.float16),
             'Precision': np.mean(outer_precision, dtype=np.float16),
             'Precisionstd': np.std(outer_precision, dtype=np.float16),
             'Recall': np.mean(outer_recall, dtype=np.float16), 'Recallstd': np.std(outer_recall, dtype=np.float16),
             'AUC': np.mean(outer_auc, dtype=np.float16), 'AUCstd': np.std(outer_auc, dtype=np.float16),
             'PRAUC': np.mean(outer_pr_auc, dtype=np.float16), 'PRAUCstd': np.std(outer_pr_auc, dtype=np.float16),
             'Specificity': specificity, 'Sensitivity': sensitivity})
    ##
