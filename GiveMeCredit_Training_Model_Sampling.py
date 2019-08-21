import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc


from functions_library import data_collection, nan_detector, nan_rmv_indication, nan_replacement, feature_dist_plot,\
                outlier_detector, feature_augmentation, RF_Classifier_func, GB_Classifier_func, \
                SMOTE_oversampling, plotting_AUC_Bestestimator

# Reading data from the data sets provided.
[train_data,test_data,train_labels,test_labels,feature_list] = data_collection()
# Detecting NaN values in different features in both training and test sets.
[train_nan_features,train_an_cnt] = nan_detector(train_data, feature_list)
[test_nan_features,test_nan_cnt] = nan_detector(test_data, feature_list)
# Generating new feature columns that indicate the existence of NaN values for each entry,
# and data samples with NaN values are removed for the sake of plotting the distribution.
[train_data,train_data_nan_drop] = nan_rmv_indication(train_data, train_nan_features)
[test_data, test_data_nan_drop] = nan_rmv_indication(test_data, test_nan_features)
# Plotting the distribution of each of the features in both linear and log scales.
feature_distribution_plot = 0
if feature_distribution_plot == 1:
    feature_dist_plot(train_data,train_data_nan_drop,feature_list, train_nan_features)


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


# Training the RF classifier and the GB classifier models.
[RF_Predicts_df,best_RF_Classifier] = RF_Classifier_func(smotesampled_X_df, smotesampled_Y_df, test_X)
[GB_Predicts_df,best_GB_Classifier] = GB_Classifier_func(smotesampled_X_df, smotesampled_Y_df, test_X)


# Plotting the performance of the RF and GB classifiers over the entire training set.
class_indicator_str = 'Random Forest Classifier'
pred = best_RF_Classifier.predict_proba(train_X)
fpr, tpr, thresholds = roc_curve(train_Y, pred[:, 1])
roc_auc = auc(fpr, tpr)
plotting_AUC_Bestestimator(fpr, tpr, roc_auc, class_indicator_str)

class_indicator_str = 'Gradient Boosting Classifier'
pred = best_GB_Classifier.predict_proba(train_X)
fpr, tpr, thresholds = roc_curve(train_Y, pred[:, 1])
roc_auc = auc(fpr, tpr)
plotting_AUC_Bestestimator(fpr, tpr, roc_auc, class_indicator_str)
