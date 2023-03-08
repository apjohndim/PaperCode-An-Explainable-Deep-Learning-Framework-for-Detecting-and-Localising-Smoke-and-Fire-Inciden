# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 13:24:17 2022

@author: User
"""
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report


def metrics (predictions_all, predictions_all_num, predictions_all_cat, labels, verbose):
    
    recall = recall_score(labels[:,1],predictions_all)   
    precision = precision_score(labels[:,1],predictions_all)
    aucS = roc_auc_score(labels[:,1], predictions_all)
    conf = confusion_matrix(labels[:,1], predictions_all) #get the fold conf matrix
    f1 = f1_score(labels[:,1], predictions_all)
    
    FP = conf[0,1]
    FN = conf[1,0]
    TP = conf[1,1]
    TN = conf[0,0]
    
    specificity = TN/(TN+FP)
    sensitivity = TP/(TP+FN)
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    cohen = cohen_kappa_score(labels[:,1],predictions_all)

    report = classification_report(labels[:,1],predictions_all)
    
    pr, re, thresholds = precision_recall_curve(labels[:,1],predictions_all)
    
    metric_results = {
               'Accuracy'     : round(accuracy,2),
               'Precision'    : round(precision,2),
               'Recall'       : round(recall,2),
               'AUC'          : round(aucS,2),
               'F1'           : round(f1,2),
               'TP'           : TP,
               'FP'           : FP,
               'TN'           : TN,
               'FN'           : FN,
               'Sensitivity'  : round(sensitivity,2),
               'Specificity'  : round(specificity,2),
               'PPV'          : round(PPV,2),
               'NPV'          : round(NPV,2),
               'Cohen'        : cohen,
               'CM'           : conf,
               'Report'       : report,
               'pr variable'  : pr,
               'rc variable'  : re,
               'thresholds'   : thresholds
                        }
    
    if verbose:
        k_list = list(metric_results.keys())
        for i in range (14):
            print ( '{} : {}'.format(k_list[i],metric_results[k_list[i]]))
    return metric_results
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    