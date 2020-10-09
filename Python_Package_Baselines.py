#%%
import random
predicted_labels = random.choices((0,1), k = 10000, weights = (0.99, 0.1))
true_labels = random.choices((0,1), k = 10000, weights = (0.99, 0.1))
theta = 0.5
measure = 'PPV'
# def basic_baseline(predicted_labels, true_labels, measure = ('TP', 'TN', 'FN', 'FP', 'TPR', 'NPR'), beta = 1):
    # add more measures
 
# %%
import statistics
import math
from scipy.stats import hypergeom
P = sum(true_labels)
M = len(true_labels)
N = M - P
return_statistics = {}
if (measure.upper() in ['TP']):
    return_statistics['Max Expected Value'] = P
    return_statistics['Argmax Expected Value'] = 1
    return_statistics['Min Expected Value'] = 0
    return_statistics['Argmin Expected Value'] = 0

if (measure.upper() in ['TN']):
    return_statistics['Max Expected Value'] = N
    return_statistics['Argmax Expected Value'] = 0
    return_statistics['Min Expected Value'] = 0
    return_statistics['Argmin Expected Value'] = 1

if (measure.upper() in ['TPR']):
    return_statistics['Max Expected Value'] = 1
    return_statistics['Argmax Expected Value'] = 1
    return_statistics['Min Expected Value'] = 0
    return_statistics['Argmin Expected Value'] = 0

if (measure.upper() in ['TNR']):
    return_statistics['Max Expected Value'] = 1
    return_statistics['Argmax Expected Value'] = 0
    return_statistics['Min Expected Value'] = 0
    return_statistics['Argmin Expected Value'] = 1

if (measure.upper() in ['PPV']):
    return_statistics['Max Expected Value'] = P/M
    return_statistics['Argmax Expected Value'] = [i/M for i in range(1, M + 1)]
    return_statistics['Min Expected Value'] = P/M
    return_statistics['Argmin Expected Value'] = [i/M for i in range(1, M + 1)]

if (measure.upper() in ['NPV']):
    return_statistics['Max Expected Value'] = N/M
    return_statistics['Argmax Expected Value'] = [i/M for i in range(0, M)]
    return_statistics['Min Expected Value'] = N/M
    return_statistics['Argmin Expected Value'] = [i/M for i in range(0, M)]

if (measure.upper() in ['FDR']):
    return_statistics['Max Expected Value'] = N/M
    return_statistics['Argmax Expected Value'] = [i/M for i in range(1, M + 1)]
    return_statistics['Min Expected Value'] = N/M
    return_statistics['Argmin Expected Value'] = [i/M for i in range(1, M + 1)]

if (measure.upper() in ['FOR']):
    return_statistics['Max Expected Value'] = P/M
    return_statistics['Argmax Expected Value'] = [i/M for i in range(0, M)]
    return_statistics['Min Expected Value'] = P/M
    return_statistics['Argmin Expected Value'] = [i/M for i in range(0, M)]

if (measure.upper() in ['ACC', 'ACCURACY']):
    return_statistics['Max Expected Value'] = max((N/M, P/M))
    return_statistics['Argmax Expected Value'] = int((P >= N))
    return_statistics['Min Expected Value'] = min((N/M, P/M))
    return_statistics['Argmin Expected Value'] = int((P < N))

if (measure.upper() in ['BACC', 'BALANCED ACCURACY']):
    return_statistics['Max Expected Value'] = max((N/M, P/M))
    return_statistics['Argmax Expected Value'] = int((P >= N))
    return_statistics['Min Expected Value'] = min((N/M, P/M))
    return_statistics['Argmin Expected Value'] = int((P < N))

if (measure.upper() in ['FBETA', 'FSCORE', 'F', 'F BETA', 'F BETA SCORE', 'FBETA SCORE']):
    beta_squared = beta ^ 2
    return_statistics['Max Expected Value'] = (1 + beta_squared) * P / (beta_squared * P + M)
    return_statistics['Argmax Expected Value'] = 1
    return_statistics['Min Expected Value'] = 0
    return_statistics['Argmin Expected Value'] = 0

if (measure.upper() in ['MCC', 'MATTHEW', 'MATTHEWS CORRELATION COEFFICIENT']):
    return_statistics['Max Expected Value'] = 0
    return_statistics['Argmax Expected Value'] = [i/M for i in range(1, M)]
    return_statistics['Min Expected Value'] = 0
    return_statistics['Argmin Expected Value'] = [i/M for i in range(1, M)]

if (measure.upper() in ['BM', 'BOOKMAKER INFORMEDNESS', 'INFORMEDNESS']):
    return_statistics['Max Expected Value'] = 0
    return_statistics['Argmax Expected Value'] = [i/M for i in range(0, M + 1)]
    return_statistics['Min Expected Value'] = 0
    return_statistics['Argmin Expected Value'] = [i/M for i in range(0, M + 1)]

if (measure.upper() in ['MARKEDNESS', 'MK']):
    return_statistics['Max Expected Value'] = 0
    return_statistics['Argmax Expected Value'] = [i/M for i in range(1, M)]
    return_statistics['Min Expected Value'] = 0
    return_statistics['Argmin Expected Value'] = [i/M for i in range(1, M)]

if (measure.upper() in ['COHEN', 'COHENS KAPPA', 'KAPPA']):
    return_statistics['Max Expected Value'] = 0
    return_statistics['Argmax Expected Value'] = [i/M for i in range(0, M + 1)]
    return_statistics['Min Expected Value'] = 0
    return_statistics['Argmin Expected Value'] = [i/M for i in range(0, M + 1)]

if (measure.upper() in ['GMEAN1', 'G MEAN 1', 'G1']):
    return_statistics['Max Expected Value'] = math.sqrt(P)
    return_statistics['Argmax Expected Value'] = 1
    return_statistics['Min Expected Value'] = math.sqrt(P/M)
    return_statistics['Argmin Expected Value'] = 1/M

if (measure.upper() in ['GMEAN2', 'G MEAN 2', 'G2']):
    #return_statistics['Max Expected Value'] = 1/2
    return_statistics['Argmax Expected Value'] = statistics.median([i/M for i in range(0, M + 1)])
    return_statistics['Min Expected Value'] = 0
    return_statistics['Argmin Expected Value'] = 0

if (measure.upper() in ['FOWLKES-MALLOWS', 'FOWLKES MALLOWS', 'FOWLKES', 'MALLOWS']):
    return_statistics['Max Expected Value'] = math.sqrt(P / M)
    return_statistics['Argmax Expected Value'] = 1
    return_statistics['Min Expected Value'] = math.sqrt(P) / M
    return_statistics['Argmin Expected Value'] = 1/M


if (measure.upper() in ['THREAT SCORE', 'CRITICAL SUCCES INDEX', 'TS', 'CSI']):
    return_statistics['Max Expected Value'] = P / M
    return_statistics['Argmax Expected Value'] = 1
    return_statistics['Min Expected Value'] = P / (P * (M - 1) + M)
    return_statistics['Argmin Expected Value'] = 1 / M




# %%
return_statistics
# %%
rounded_m_theta = int(round(M * theta))
rv = hypergeom(M = M, n = P, N = rounded_m_theta)
x = range(0, M + 1)
result = rv.pmf(x)
# %%
import matplotlib.pyplot as plt
plt.plot(x, result)
# %%


# %%
import numpy as np
# %%
max(result)
# %%
# Kijken of we de PT goed kunnen bepalen met hetzelfde idee
result = [np.nan] * (M + 1)
for i in range(1, M):
    theta = i / M
    rounded_m_theta = int(round(M * theta))
    TP_rv = hypergeom(M = M, n = P, N = rounded_m_theta)
    result[i] = sum([((math.sqrt((k / P) * (-(k - rounded_m_theta) / N)) + ((k - rounded_m_theta) / N)) / ((k / P) + ((k - rounded_m_theta) / N))) * TP_rv.pmf(k) if ((k / P) + ((k - rounded_m_theta) / N)) != 0 else 0 for k in range(0, min((P + 1, rounded_m_theta + 1)))])

# %%
plt.plot(np.arange(0, 1 + 1/M, 1/M), result)
# %%
max(result)
# %%
result.index(max(result))
# %%
def pmf_Y(y, a, b, M, P, rounded_m_theta):
    TP_rv = hypergeom(M = M, n = P, N = rounded_m_theta)
    return(TP_rv.pmf((y - b) / a))
# %%
# Kijken of we de G2 goed kunnen bepalen met hetzelfde idee
result = [np.nan] * (M + 1)
for i in range(0, M + 1):
    theta = i / M
    rounded_m_theta = int(round(M * theta))
    TP_rv = hypergeom(M = M, n = P, N = rounded_m_theta)
    result[i] = sum([(math.sqrt(k * (N - rounded_m_theta + k)/ (P * N))) * TP_rv.pmf(k) if TP_rv.pmf(k) > 0 else 0 for k in range(0, min((P + 1, rounded_m_theta + 1)))])
# %%
plt.plot(np.arange(0, 1 + 1/M, 1/M), result)
# %%
result.index(max(result))
# %%
max_size = 100
result_array_max = np.zeros((max_size + 1, max_size + 1))
result_array_max_index = np.zeros((max_size + 1, max_size + 1))
for P in range(1, max_size + 1):
    print ('P:', P)
    for N in range(1, max_size + 1):
        M = P + N
        result = [np.nan] * (M + 1)
        for i in range(0, M + 1):
            theta = i / M
            rounded_m_theta = int(round(M * theta))
            TP_rv = hypergeom(M = M, n = P, N = rounded_m_theta)
            result[i] = sum([(math.sqrt(k * (N - rounded_m_theta + k)/ (P * N))) * TP_rv.pmf(k) if TP_rv.pmf(k) > 0 else 0 for k in range(0, min((P + 1, rounded_m_theta + 1)))])
        result_array_max_index[P, N] = result.index(max(result)) / M
        result_array_max[P,N] = max(result)
# %%
import pickle
# with open ('results_gmean2.pkl', 'wb') as output:
#     pickle.dump([result_array_max, result_array_max_index], output)
with open ('results_gmean2.pkl', 'rb') as input:
    result_array_max, result_array_max_index = pickle.load(input)
# %%
import gplearn
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.random import check_random_state
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import graphviz
# %%
%pylab inline
constant_value = 20
constant_value_2 = 3
x0 = np.arange(1, constant_value + 1, 1)
x1 = np.arange(1, constant_value_2 + 1, 1)
mesh_x0, mesh_x1 = np.meshgrid(x0, x1)
y_truth = result_array_max[mesh_x0, mesh_x1]

# %%
# Training samples
mesh_x0_reshaped = mesh_x0.reshape(prod(mesh_x0.shape))
mesh_x1_reshaped = mesh_x1.reshape(prod(mesh_x1.shape))
X_train = np.column_stack((mesh_x0_reshaped, mesh_x1_reshaped)) 
y_train = result_array_max[X_train[:, 0], X_train[:, 1]]

def _factorial_function(x1):
    results = np.zeros(len(x1))
    for i,k in enumerate(x1):
        if k >= 0 and k < 100:
            results[i] = math.factorial(int(k))
    return(results)

factorial_function = gplearn.functions.make_function(function = _factorial_function, name = 'factorial_function', arity = 1)


# %%

est_gp = SymbolicRegressor(population_size=5000,
                           generations=100, stopping_criteria=0.00001,
                           verbose = 1,
                           parsimony_coefficient = 0.0005,
                           metric = 'mean absolute error',
                           low_memory= True,
                           feature_names = ["P", "N"],
                           function_set = ('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min', factorial_function))
est_gp.fit(X_train, y_train)
print(est_gp._program)
# %%

# %%
