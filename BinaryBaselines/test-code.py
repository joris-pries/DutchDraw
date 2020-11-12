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

sb. 
sb.measure_score(y_true, y_pred, measure)