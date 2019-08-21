import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV


def data_collection():
    '''
    This function reads labeled training data and test data from the .csv files provided.
    :return: List of features, training data, test data, and the corresponding labels for each of those data sets.
    '''
    file_address = 'Data'
    train_data = pd.read_csv(file_address+'/cs-training.csv')
    test_data = pd.read_csv(file_address+'/cs-test.csv')
    test_y_data = pd.read_csv(file_address+'/sampleEntry.csv')
    train_data.columns = train_data.columns.values
    test_data.columns = test_data.columns.values
    feature_list = list(train_data.columns.values)
    train_labels = train_data['SeriousDlqin2yrs']
    test_labels = test_y_data['Probability']
    # Removing the first two columns that are not 'features'
    feature_list.remove('Unnamed: 0')
    feature_list.remove('SeriousDlqin2yrs')
    # ax = sns.countplot(train_labels)  # Plotting distribution of labels to see if data is balanced or imbalanced
    return train_data,test_data,train_labels,test_labels,feature_list


def nan_detector(data, feature_list):
    '''
    This function detects the features that include NaN contents in both training data set and test data set,
    and outputs those features with the number of NaN values in each feature.
    :param data: is the training or test data over which NaN values are searched for.
    :param feature_list: the list of features extracted from the data sheet provided above.
    :return: nan_features, which are the features with NaN values, and nan_cnt that counts the number of NaN values.
    '''
    nan_cnt = {}
    nan_features = []
    for c in feature_list:
        nan_val = 0
        for val in data[c].isnull():
            if val == True:
                nan_val = nan_val + 1
                if nan_val == 1:
                    nan_features.append(c)
        nan_cnt[c] = [nan_val]

    return nan_features, nan_cnt


def nan_rmv_indication(data, nan_features):
    '''
    This function generates a new data frame with all NaN samples removed, 'data_nan'. This new data frame will be used
    later to obtain distribution of the samples.
    The function also creates extra columns/features which are named after the features with NaN values, e.g.,
    'feature name'_nans. Content 'i' of the newly added 'feature name'_nans is 1 if content 'i' of 'feature name' is NaN,
    and it is 0 otherwise. This column indicates the existence of NaN value in the corresponding index which may
    be informative in training the final model.
    :param data: training or test data provided.
    :param nan_features: the list of features that contain NaN values.
    :return: data with extra columns/features that indicate the existence of NaN values,
    and a data frame in which NaN values are removed.
    '''
    data_nan = pd.DataFrame()
    for feature in nan_features:
        data[str(feature) + '_nans'] = pd.isnull(data[[feature]]) * 1
    data_nan = data.dropna(how='any')
    return data, data_nan


def nan_replacement(data, nan_features):
    '''
    This function replaces NaN values of each feature with the median of the values in that feature.
    Considering the distribution of the provided data, which seems to have few numbers of outliers,
    median would be a good choice to be used for NaN values.
    :param data: training or test data provided.
    :param nan_features: the list of features that contain NaN values.
    :return: data with NaN values replaced by median of the contents of corresponding feature.
    '''
    for feature in nan_features:
        data[feature] = data[feature].fillna(data[feature].median())
    return data


def feature_dist_plot(data, data_nan_drop, feature_list, nan_features):
    '''
    This function generates different plots each of which illustrating the distribution of either the contents
    or log scale of the contents of each of the features. For the features with skewed distribution of the contents
    the log scale of the contents can be used.
    (For convenience, the distributions in both cases are provided for each feature.)
    :param data: training or test data provided.
    :param data_nan_drop: training or test data in which NaN values are all dropped (without replacing with any new value)
    :param feature_list: the list of features.
    :param nan_features: the list of features that contain NaN values.
    :return: different plots each of which showing the distribution of the data or log scale of data.
    '''
    data_log = pd.DataFrame()
    data_nan_drop_log = pd.DataFrame()
    for features in feature_list:
        if features in nan_features:
            data_nan_drop_log[features] = np.log(1 + data_nan_drop[features].values)
            sns.distplot(data_nan_drop[features])
            plt.show()
            sns.distplot(data_nan_drop_log[features])
            plt.show()
        else:
            data_log[features] = np.log(1 + data[features].values)
            sns.distplot(data[features])
            plt.show()
            sns.distplot(data_log[features])
            plt.show()


def outlier_detector(data, feature_list):
    for feature in feature_list:
        print(feature)
        data_Q1 = data[feature].quantile(0.25)
        data_Q3 = data[feature].quantile(0.75)
        IQR = data_Q3-data_Q1
        lower_bnd = data_Q1 - 1.5 * IQR
        upper_bnd = data_Q3 + 1.5 * IQR
        outlier_idx = (data[feature]<lower_bnd) | (data[feature]>upper_bnd)
        data.loc[outlier_idx,feature] = data[feature].median()
    return data


def feature_augmentation(train_data,test_data):
    '''
    In order to build a more accurate and more robust model, this function generates some additional features
    based on the provided features. The "WeightedPastDue" includes a weighted sum of the number of times with past due.
    The rationale behind this is the fact that a longer past due could have more impact on the probability of delinquency.
    To capture this observation, by considering the lower-bound of each interval as a representative
    for each case, e.g., 30, 60, and 90 days, the weights are chosen to be 1,2, and 3, respectively.
    Another feature is "MonthlyNetIncome" which represents the difference between the income and the amount payed for all
     debts.
    :param train_data:
    :param test_data:
    :return: New features generated according to the existing ones
    '''
    train_data['WeightedPastDue'] = (train_data['NumberOfTimes90DaysLate'] + 2 * train_data[
        'NumberOfTime60-89DaysPastDueNotWorse'] + 3 * train_data['NumberOfTime30-59DaysPastDueNotWorse']) / 6
    test_data['WeightedPastDue'] = (test_data['NumberOfTimes90DaysLate'] + 2 * test_data[
        'NumberOfTime60-89DaysPastDueNotWorse'] + 3 * test_data['NumberOfTime30-59DaysPastDueNotWorse']) / 6
    train_data['MonthlyNetIncome'] = train_data['MonthlyIncome'] * (1 - train_data['DebtRatio'])
    test_data['MonthlyNetIncome'] = test_data['MonthlyIncome'] * (1 - train_data['DebtRatio'])

    return train_data, test_data


def RF_Classifier_func(train_X, train_Y, test_X):
    '''
    This function build a classification model based on the Random Forest classifier.
    :param train_X: train_X: training data set which is provided.
    :param train_Y: the probability of delinquency corresponding to each index.
    :param test_X: test data set which is provided.
    :return: It returns the predictions on the delinquency probabilities for the test data set.
    '''
    RF_Classifier = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,
                                           min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_leaf_nodes=None,
                                           bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, verbose=1)
    RF_Params = {'criterion': ['gini'], 'n_estimators': (100, 200, 300, 400, 500), 'max_depth': (6, 8, 10)}
    RF_Search = RandomizedSearchCV(estimator=RF_Classifier, param_distributions=RF_Params, n_iter=10, scoring='roc_auc',
                                 cv=None, verbose=2).fit(train_X, train_Y)
    RF_Predicts = RF_Search.predict_proba(test_X)
    RF_Predicts_df = pd.DataFrame(np.ndarray.tolist(RF_Predicts[:,1]), columns =['Probability'])
    # Saving the trained model
    RF_Filename = "RF_Classifier_smotesmpling.pkl"
    with open(RF_Filename, 'wb') as file:
        pickle.dump(RF_Search, file)
    return RF_Predicts_df,RF_Search.best_estimator_


def GB_Classifier_func(train_X, train_Y, test_X):
    '''
    This function build a classification model based on the Gradient Boosting classifier.
    :param train_X: training data set which is provided.
    :param train_Y: the probability of delinquency corresponding to each index.
    :param test_X: test data set which is provided.
    :return: It returns the predictions on the delinquency probabilities for the test data set.
    '''
    GB_Classifier = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0,
                                               min_samples_split=2, min_samples_leaf=1,
                                               min_weight_fraction_leaf=0.0, max_depth=3, init=None,
                                               random_state=None, max_features=None, verbose=1)
    GB_Params = {'loss': ['deviance'], 'n_estimators': (100, 200, 300, 400, 500), 'max_depth': (6, 8, 10)}
    GB_Search = RandomizedSearchCV(estimator=GB_Classifier, param_distributions=GB_Params, n_iter=10, scoring='roc_auc',
                                    cv=None, verbose=2).fit(train_X, train_Y)
    GB_Predicts = GB_Search.predict_proba(test_X)
    GB_Predicts_df = pd.DataFrame(np.ndarray.tolist(GB_Predicts[:,1]), columns =['Probability'])

    # Saving the trained model
    GB_Filename = "GB_Classifier_smotesmpling.pkl"
    with open(GB_Filename, 'wb') as file:
        pickle.dump(GB_Search, file)
    return GB_Predicts_df,GB_Search.best_estimator_


def SMOTE_oversampling(train_X, train_Y):
    smote_sampling = SMOTE(random_state=0)
    smotesampled_X, smotesampled_Y = smote_sampling.fit_resample(train_X, train_Y)
    return smotesampled_X, smotesampled_Y


def auc_roc_vars(classifier, train_X, train_Y, nfolds):
    '''
     This function generates the parameters required for plotting the performance of the models (classifiers) in terms
     of the ROC and AUC area (K-fold cross validation).
    :param classifier: the model which is used to predict the output labels (Random Forest and Gradient Boosting in this
     code).
    :param train_X: purified training data; NaN values and outliers are replaced by medians.
    :param train_Y: corresponding labels that are provided.
    :param nfolds: the desired number of folds.
    :return: the parameters required to plot the ROC curve of each folding step for the desired classifier.
    '''
    i = 0
    d_fold = KFold(nfolds, shuffle=True)
    fpr_vec = {0: 0}
    tpr_vec = {0: 0}
    roc_auc_vec = {0: 0}
    for KFold_train, KFold_test in d_fold.split(train_X, train_Y):
        i = i + 1
        predics_kfold = classifier.fit(train_X.iloc[KFold_train], train_Y.iloc[KFold_train]).predict_proba(
            train_X.iloc[KFold_test])
        fpr, tpr, thresholds = roc_curve(train_Y.iloc[KFold_test], predics_kfold[:, 1])
        roc_auc = auc(fpr, tpr)
        fpr_vec[i-1] = fpr
        tpr_vec[i-1] = tpr
        roc_auc_vec[i-1] = roc_auc
    return fpr_vec, tpr_vec, roc_auc_vec


def plotting_Kfold(fpr_vec, tpr_vec, roc_auc_vec, nfolds, class_indicator):
    for i in range(nfolds):
        fpr, tpr, thresholds = [fpr_vec[i], tpr_vec[i], roc_auc_vec[i]]
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %f)' % (i, roc_auc))

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('CV ROC curve %s' % class_indicator)
    plt.legend(loc="lower right")
    fig = plt.gcf()
    fig.set_size_inches(15, 5)
    plt.show()

def plotting_AUC_Bestestimator(fpr, tpr, roc_auc, class_indicator):
    plt.plot(fpr, tpr, lw=1, label='ROC fold (area = %f)' % (roc_auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('CV ROC curve %s' % class_indicator)
    plt.legend(loc="lower right")
    fig = plt.gcf()
    fig.set_size_inches(15, 5)
    plt.show()