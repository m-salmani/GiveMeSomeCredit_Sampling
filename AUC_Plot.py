import pickle
from functions_library import plotting_Kfold

# Plotting AUC/ROC curve for the RF classifier.
[fpr_vec, tpr_vec, roc_auc_vec,nfolds,class_indicator]= pickle.load(open('var_final_plot_RF.pkl', 'rb'))
plotting_Kfold(fpr_vec, tpr_vec, roc_auc_vec, nfolds, class_indicator)

# Plotting AUC/ROC curve for the GB classifier.
[fpr_vec, tpr_vec, roc_auc_vec,nfolds,class_indicator]= pickle.load(open('var_final_plot_GB.pkl', 'rb'))
plotting_Kfold(fpr_vec, tpr_vec, roc_auc_vec, nfolds, class_indicator)

