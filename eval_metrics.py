import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# forecasting
def Mean_Squared_Error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def Root_Mean_Squared_Error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def Mean_Absolute_Error(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def Mean_Absolute_Percentage_Error(y_true, y_pred):
    return np.sum(np.abs((y_true - y_pred)/y_true)/len(y_true)) * 100

def Symmetric_Mean_Absolute_Percentage_Error(y_true, y_pred):
    return np.sum(np.abs(y_true-y_pred)/( (np.abs(y_true) + np.abs(y_pred))/2.0))/len(y_true) * 100

def Root_Relative_Squared_Error(y_true, y_pred, epsilon=1e-12):
    rse_numerator = np.sum(np.subtract(y_pred, y_true) ** 2)
    rse_demominator = np.sum(np.subtract(y_true, np.mean(y_true)) ** 2)
    rres = np.sqrt(np.divide(rse_numerator, rse_demominator + epsilon))
    return rres

def Empirical_Correlation_Coefficient(y_true, y_pred, epsilon=1e-12):
    y_true_mean = np.mean(y_true, axis=0)
    y_pred_mean = np.mean(y_pred, axis=0)

    numerator = np.sum((y_true - y_true_mean) * (y_pred - y_pred_mean), axis=0)
    denominator_true = np.sqrt(np.sum((y_true - y_true_mean)**2, axis=0))
    denominator_pred = np.sqrt(np.sum((y_pred - y_pred_mean)**2, axis=0))
    denominator = denominator_true * denominator_pred 
    corr = np.mean(np.divide(numerator, denominator + epsilon))
    return corr


# classification
def Accuracy_Score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def Precision_Score(y_true, y_pred):
    return precision_score(y_true, y_pred, average='macro')

def Recall_Score(y_true, y_pred):
    return recall_score(y_true, y_pred, average='macro')

def F1_Score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def Auc_Score(y_true, y_pred):
    return roc_auc_score(y_true, y_pred, average='macro')

