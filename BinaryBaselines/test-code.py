# -*- coding: utf-8 -*-
"""
@author: Etienne van de Bijl
"""
import numpy as np
import datetime
from sklearn.metrics import (confusion_matrix, fbeta_score, matthews_corrcoef, 
                             accuracy_score, cohen_kappa_score, balanced_accuracy_score)
import shuffle_baseline as sb

y_true = np.random.randint(2, size = 1000)
y_pred = np.random.randint(2, size = 1000)

# %%
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

#Furthermore, we miss a value error on the beta value when beta is smaller than 0

# %%

times = []
sizes = [10,50,100,500,1000,5000]
for size in sizes:
    y_true = np.random.randint(2, size = size)
    start = datetime.datetime.now()
    sb.optimized_basic_baseline(y_true.tolist(), "G2", beta=1)
    time = (datetime.datetime.now() - start).total_seconds()
    print(time)
    times.append(time)
    
import matplotlib.pyplot as plt    
plt.plot(sizes, times)    

# %%

y_true = np.random.randint(2, size = 20)
for key, value in sb.name_dictionary.items():
    result = sb.optimized_basic_baseline(y_true.tolist(), key)
    print(key)
    print(result)

sb.optimized_basic_baseline(y_true.tolist(), 'FBETA', -4)
    
"""
Dingen met Joris te bespreken:
    - Vervangen van check_targets (als dingen niet zelfde lengte zijn bijvoorbeeld)
    - Ervoor zorgen dat numpy ook geaccepteerd wordt, alleen lijst is irritant
    - Beta afdwingen dat het hoger moet zijn dan 0
    - Tijd tot uitrekenen van G2 bijvoorbeeld duurt enorm lang.
    - De uitleg van basic_baseline mag iets duidelijker want ik begrijp niet hoe ik de functie moet gebruiken.
"""

# %%

generator = sb.basic_baseline(y_true.tolist(), "TP", beta=1)
print(generator)
generator.keys()
generator["Expectation Function"](0.5)





    