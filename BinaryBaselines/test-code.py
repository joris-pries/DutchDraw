# -*- coding: utf-8 -*-
"""
@author: Etienne van de Bijl
"""
import numpy as np
from sklearn.metrics import (confusion_matrix, fbeta_score, matthews_corrcoef, 
                             accuracy_score, cohen_kappa_score, balanced_accuracy_score)
import shuffle_baseline as sb

y_true = np.random.randint(2, size = 100)
y_pred = np.random.randint(2, size = 100)

#Something disturbing happens: We have to cast the np array to a list. This is 
#risky.
sb.measure_score(y_true.tolist(), y_pred.tolist(), "TP")
sb.measure_score(y_true.tolist(), y_pred.tolist(), "TN")
sb.measure_score(y_true.tolist(), y_pred.tolist(), "FP")
sb.measure_score(y_true.tolist(), y_pred.tolist(), "FN")

confusion_matrix(y_true, y_pred)

sb.measure_score(y_true.tolist(), y_pred.tolist(), "F")
fbeta_score(y_true, y_pred, 1)

sb.measure_score(y_true.tolist(), y_pred.tolist(), "MCC")
matthews_corrcoef(y_true, y_pred)

sb.measure_score(y_true.tolist(), y_pred.tolist(), "ACC")
accuracy_score(y_true, y_pred)

sb.measure_score(y_true.tolist(), y_pred.tolist(), "BACC")
balanced_accuracy_score(y_true, y_pred)

cohen_kappa_score(y_true,y_pred)
sb.measure_score(y_true.tolist(), y_pred.tolist(), "COHEN")

fbeta_score([1,0,1],[1,1,1],1)

#Error
sb.measure_score([1,1,0], [1,0,1,0], "TP")

#The best way to avoid this problem is to use the class _check_targets(y_true, y_pred) in the _classification.py file of sklearn.metrics
from sklearn.metrics._classification import _check_targets
_check_targets(y_true, y_pred)

