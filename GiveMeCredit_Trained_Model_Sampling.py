import pandas as pd
import numpy as np
import pickle

from functions_library import data_collection, nan_detector, nan_rmv_indication, nan_replacement, outlier_detector, \
                              feature_augmentation, SMOTE_oversampling, auc_roc_vars


# Reading data from the data sets provided.
[train_data,test_data,train_labels,test_labels,feature_list] = data_collection()
# Detecting NaN values in different features in both training and test sets.
[train_nan_features,train_an_cnt] = nan_detector(train_data, feature_list)
[test_nan_features,test_nan_cnt] = nan_detector(test_data, feature_list)
# Generating new feature columns that indicate the existence of NaN values for each entry,
# and data samples with NaN values are removed for the sake of plotting the distribution.
[train_data, train_data_nan_drop] = nan_rmv_indication(train_data, train_nan_features)
[test_data, test_data_nan_drop] = nan_rmv_indication(test_data, test_nan_features)


# For features with skewed distributions, log-scale distribution is provided to have better insights.
log_feature_list = ['RevolvingUtilizationOfUnsecuredLines', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
                    'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse','MonthlyIncome']
for l_feature in log_feature_list:
    train_data[l_feature] = np.log(1 + train_data[l_feature].values)
    test_data[l_feature] = np.log(1 + test_data[l_feature].values)


# Data Pre-processing:
# Replacing NaN values with the median in both training and test sets.
train_data = nan_replacement(train_data, train_nan_features)
test_data = nan_replacement(test_data, test_nan_features)
# Detecting outliers and replacing them with the median in training data set.
train_data = outlier_detector(train_data, feature_list)
# Generating some new features based on the existing ones in both training and test sets.
train_data, test_data = feature_augmentation(train_data,test_data)


# Gathering all the training data, training labels, test data, and test labels.
train_X = train_data.drop(['SeriousDlqin2yrs', 'Unnamed: 0'], axis=1, inplace=False)
train_Y = train_labels
test_X = test_data.drop(['SeriousDlqin2yrs', 'Unnamed: 0'], axis=1, inplace=False)
test_Y = test_labels


# Using SMOTE for over-sampling the minority class to balance the imbalanced data set.
[smotesampled_X, smotesampled_Y] = SMOTE_oversampling(train_X, train_Y)
smotesampled_X_df = pd.DataFrame(smotesampled_X)
smotesampled_Y_df = pd.DataFrame(smotesampled_Y)


# Loading the trained models and plotting ROC curve for the RF classifier.
nfolds_number = 5
trained_RF_model = pickle.load(open('RF_Classifier_smotesmpling.pkl', 'rb'))
class_indicator_str = 'Random Forest Classifier'
fpr_vec, tpr_vec, roc_auc_vec = auc_roc_vars(trained_RF_model.best_estimator_, smotesampled_X_df, smotesampled_Y_df, nfolds=nfolds_number)
VAR_RF_Filename = "var_final_plot_RF.pkl"
with open(VAR_RF_Filename, 'wb') as file:
    pickle.dump([fpr_vec, tpr_vec, roc_auc_vec,nfolds_number,class_indicator_str], file)


# Loading the trained models and plotting ROC curve for the GB classifier.
nfolds_number = 5
trained_GB_model = pickle.load(open('GB_Classifier_smotesmpling.pkl', 'rb'))
class_indicator_str = 'Gradient Boosting Classifier'
fpr_vec, tpr_vec, roc_auc_vec = auc_roc_vars(trained_GB_model.best_estimator_, smotesampled_X_df, smotesampled_Y_df, nfolds=nfolds_number)
VAR_GB_Filename = "var_final_plot_GB.pkl"
with open(VAR_GB_Filename, 'wb') as file:
    pickle.dump([fpr_vec, tpr_vec, roc_auc_vec,nfolds_number,class_indicator_str], file)



