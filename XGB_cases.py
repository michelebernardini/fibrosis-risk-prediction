import pandas as pd
import numpy as np
from numpy.random import seed
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, auc, precision_recall_curve
from collections import Counter
import xgboost as xgb
from imblearn.over_sampling import SMOTE


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

        parameters = {'n_estimators': [75, 100, 150, 200],
                      'max_depth': [6, 25, 50, 75],
                     'eta': [0.05, 0.1, 0.2, 0.3]}

        model = GridSearchCV(estimator=classifier, param_grid=[parameters], cv=skf_int, scoring='recall_macro')
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

    print('Accuracy: %.4f' % (np.mean(outer_accuracy)))
    print('F1: %.4f' % (np.mean(outer_f1)))
    print('Precision: %.4f' % (np.mean(outer_precision)))
    print('Recall: %.4f' % (np.mean(outer_recall)))
    print('AUC: %.4f' % (np.mean(outer_auc)))
    print('PR_AUC: %.4f' % (np.mean(outer_pr_auc)))
    print(cm_tot)
    ###
