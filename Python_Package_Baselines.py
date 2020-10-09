#%%
import random
predicted_labels = random.choices((0,1), k = 10000, weights = (0.99, 0.1))
true_labels = random.choices((0,1), k = 10000, weights = (0.99, 0.1))
theta = 0.5
measure = 'PT'
# def basic_baseline(predicted_labels, true_labels, measure = ('TP', 'TN', 'FN', 'FP', 'TPR', 'NPR'), beta = 1):
    # add more measures
 
# %%
import statistics
import math
from scipy.stats import hypergeom
import numpy as np

def optimized_basic_baseline(true_labels, measure = ('TP', 'TN', 'FN', 'FP', 'TPR', 'NPR', 'PPV', 'NPV', 'FDR', 'FOR', 'ACC', 'BACC', 'FBETA', 'MCC', 'BM', 'MK', 'COHENS KAPPA', 'GMEAN1', 'GMEAN2', 'GMEAN2 APPROX', 'FOWLKES MALLOWS', 'TS', 'PT'), beta = 1):
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
        beta_squared = beta ** 2
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
        print("This baseline has to be calculated. This could take some time for large M. For an approximation: use GMEAN2 APPROX")
        result = [np.nan] * (M + 1)
        for i in range(0, M + 1):
            theta = i / M
            rounded_m_theta = int(round(M * theta))
            TP_rv = hypergeom(M = M, n = P, N = rounded_m_theta)
            result[i] = sum([(math.sqrt(k * (N - rounded_m_theta + k)/ (P * N))) * TP_rv.pmf(k) if TP_rv.pmf(k) > 0 else 0 for k in range(0, min((P + 1, rounded_m_theta + 1)))])
        return_statistics['Max Expected Value'] = np.nanmax(result)
        return_statistics['Argmax Expected Value'] = [i/M for i, j in enumerate(result) if j == return_statistics['Max Expected Value'] ]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [0, 1]


    if (measure.upper() in ['GMEAN2 APPROX', 'G MEAN 2 APPROX', 'G2 APPROX']):
        print("Note: the max and argmax are an approximation.")
        return_statistics['Max Expected Value'] = 1/2
        return_statistics['Argmax Expected Value'] = 1/2
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [0, 1]



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


    if (measure.upper() in ['PREVALENCE THRESHOLD', 'PT']):
        #print("This baseline has to be calculated. This could take some time for large M.")
        result = [np.nan] * (M + 1)
        for i in [1, M - 1]:
            theta = i / M
            rounded_m_theta = int(round(M * theta))
            TP_rv = hypergeom(M = M, n = P, N = rounded_m_theta)
            result[i] = sum([((math.sqrt((k / P) * (-(k - rounded_m_theta) / N)) + ((k - rounded_m_theta) / N)) / ((k / P) + ((k - rounded_m_theta) / N))) * TP_rv.pmf(k) if ((k / P) + ((k - rounded_m_theta) / N)) != 0 else 0 for k in range(0, min((P + 1, rounded_m_theta + 1)))])
        return_statistics['Max Expected Value'] = np.nanmax(result)
        return_statistics['Argmax Expected Value'] = 1 / M
        return_statistics['Min Expected Value'] = np.nanmin(result)
        return_statistics['Argmin Expected Value'] = (M - 1) / M
    
    return(return_statistics)

# %%
optimized_basic_baseline(true_labels, measure)
# %%
##################################################################################################
def basic_baseline_statistics(theta, true_labels, measure = ('TP', 'TN', 'FN', 'FP', 'TPR', 'NPR', 'PPV', 'NPV', 'FDR', 'FOR', 'ACC', 'BACC', 'FBETA', 'MCC', 'BM', 'MK', 'COHENS KAPPA', 'GMEAN1', 'GMEAN2', 'FOWLKES MALLOWS', 'TS', 'PT'), beta = 1):
    
    P = sum(true_labels)
    M = len(true_labels)
    N = M - P
    rounded_m_theta = int(theta * M)
    theta_star = rounded_m_theta / M

    var_tp = (theta_star * (1 - theta_star) * P * N) / (M - 1)
    mean_tp = theta_star * P

    return_statistics = {}

    def generate_hypergeometric_distribution(a,b):
        def pmf_Y(y):
            TP_rv = hypergeom(M = M, n = P, N = int(theta * M))
            return(TP_rv.pmf((y - b) / a))
        return(pmf_Y)

    if (measure.upper() in ['TP']):
        a = 1
        b = 0
        return_statistics['Distribution'] = generate_hypergeometric_distribution(a,b)
        return_statistics['Variance'] = (a ** 2) * var_tp
        return_statistics['Mean'] = (a * mean_tp) + b
        

    if (measure.upper() in ['TN']):
        a = 1
        b = N - rounded_m_theta        
        return_statistics['Distribution'] = generate_hypergeometric_distribution(a,b)
        return_statistics['Variance'] = (a ** 2) * var_tp    
        return_statistics['Mean'] = (a * mean_tp) + b

    if (measure.upper() in ['TPR']):
        a = 1 / P
        b = 0        
        return_statistics['Distribution'] = generate_hypergeometric_distribution(a,b)
        return_statistics['Variance'] = (a ** 2) * var_tp    
        return_statistics['Mean'] = (a * mean_tp) + b


    if (measure.upper() in ['TNR']):
        a = 1 / N
        b = N - rounded_m_theta        
        return_statistics['Distribution'] = generate_hypergeometric_distribution(a,b)
        return_statistics['Variance'] = (a ** 2) * var_tp    
        return_statistics['Mean'] = (a * mean_tp) + b


    if (measure.upper() in ['PPV']):
        a = 1 / rounded_m_theta
        b = 0
        return_statistics['Distribution'] = generate_hypergeometric_distribution(a,b)
        return_statistics['Variance'] = (a ** 2) * var_tp    
        return_statistics['Mean'] = (a * mean_tp) + b

        

    if (measure.upper() in ['NPV']):
        a = 1 / (M - rounded_m_theta)
        b = N - rounded_m_theta
        return_statistics['Distribution'] = generate_hypergeometric_distribution(a,b)
        return_statistics['Variance'] = (a ** 2) * var_tp    
        return_statistics['Mean'] = (a * mean_tp) + b

    

    if (measure.upper() in ['FDR']):
        a =  -1 / rounded_m_theta
        b = 1
        return_statistics['Distribution'] = generate_hypergeometric_distribution(a,b)
        return_statistics['Variance'] = (a ** 2) * var_tp    
        return_statistics['Mean'] = (a * mean_tp) + b

        

    if (measure.upper() in ['FOR']):
        a = -1 / (M - rounded_m_theta)
        b = 1 - N + rounded_m_theta
        return_statistics['Distribution'] = generate_hypergeometric_distribution(a,b)
        return_statistics['Variance'] = (a ** 2) * var_tp    
        return_statistics['Mean'] = (a * mean_tp) + b

        

    if (measure.upper() in ['ACC', 'ACCURACY']):
        a = 2 / M
        b = (N - rounded_m_theta) / M
        return_statistics['Distribution'] = generate_hypergeometric_distribution(a,b)
        return_statistics['Variance'] = (a ** 2) * var_tp    
        return_statistics['Mean'] = (a * mean_tp) + b

        

    if (measure.upper() in ['BACC', 'BALANCED ACCURACY']):
        a = (1 / (2 * P)) + (1 / (2 * N))
        b = (N - rounded_m_theta) / 2
        return_statistics['Distribution'] = generate_hypergeometric_distribution(a,b)
        return_statistics['Variance'] = (a ** 2) * var_tp    
        return_statistics['Mean'] = (a * mean_tp) + b

        

    if (measure.upper() in ['FBETA', 'FSCORE', 'F', 'F BETA', 'F BETA SCORE', 'FBETA SCORE']):
        beta_squared = beta ** 2
        a = (1 + beta_squared) / (beta_squared * P + M * theta_star)
        b = 0
        return_statistics['Distribution'] = generate_hypergeometric_distribution(a,b)
        return_statistics['Variance'] = (a ** 2) * var_tp    
        return_statistics['Mean'] = (a * mean_tp) + b

    

    if (measure.upper() in ['MCC', 'MATTHEW', 'MATTHEWS CORRELATION COEFFICIENT']):
        a = 1 / (math.sqrt(theta_star * (1 - theta_star) * P * N))
        b = - theta_star * P / (math.sqrt(theta_star * (1 - theta_star) * P * N))
        return_statistics['Distribution'] = generate_hypergeometric_distribution(a,b)
        return_statistics['Variance'] = (a ** 2) * var_tp    
        return_statistics['Mean'] = (a * mean_tp) + b

        

    if (measure.upper() in ['BM', 'BOOKMAKER INFORMEDNESS', 'INFORMEDNESS']):
        a = (1 / P) + (1 / N)
        b = - rounded_m_theta / N
        return_statistics['Distribution'] = generate_hypergeometric_distribution(a,b)
        return_statistics['Variance'] = (a ** 2) * var_tp    
        return_statistics['Mean'] = (a * mean_tp) + b

        

    if (measure.upper() in ['MARKEDNESS', 'MK']):
        a = (1 / rounded_m_theta) + (1 / (M - rounded_m_theta))
        b =  - rounded_m_theta / (M - rounded_m_theta)
        return_statistics['Distribution'] = generate_hypergeometric_distribution(a,b)
        return_statistics['Variance'] = (a ** 2) * var_tp    
        return_statistics['Mean'] = (a * mean_tp) + b

    

    if (measure.upper() in ['COHEN', 'COHENS KAPPA', 'KAPPA']):
        a = 2 / ((1 - theta_star) * P + theta_star * N)
        b = - 2 * theta_star * P / ((1 - theta_star) * P + theta_star * N)
        return_statistics['Distribution'] = generate_hypergeometric_distribution(a,b)
        return_statistics['Variance'] = (a ** 2) * var_tp    
        return_statistics['Mean'] = (a * mean_tp) + b

        

    if (measure.upper() in ['GMEAN1', 'G MEAN 1', 'G1']):
        a = 1 / (math.sqrt(P * rounded_m_theta))
        b = 0
        return_statistics['Distribution'] = generate_hypergeometric_distribution(a,b)
        return_statistics['Variance'] = (a ** 2) * var_tp    
        return_statistics['Mean'] = (a * mean_tp) + b

        

    if (measure.upper() in ['GMEAN2', 'G MEAN 2', 'G2']):
        def pmf_Y(y):
            TP_rv = hypergeom(M = M, n = P, N = int(theta * M))
            help_constant = math.sqrt((rounded_m_theta ** 2) - 2 * rounded_m_theta * N + (N ** 2) + 4 * P * N * (y ** 2))
            return(TP_rv.pmf((1/2) * ((- help_constant) + rounded_m_theta - N)) + TP_rv.pmf((1/2) * (help_constant + rounded_m_theta - N)))
        return_statistics['Distribution'] = pmf_Y
        # return_statistics['Variance'] = (a ** 2) * var_tp    
        # return_statistics['Mean'] = (a * mean_tp) + b

        



    if (measure.upper() in ['FOWLKES-MALLOWS', 'FOWLKES MALLOWS', 'FOWLKES', 'MALLOWS']):
        a = 1 / (math.sqrt(P * rounded_m_theta))
        b = 0
        return_statistics['Distribution'] = generate_hypergeometric_distribution(a,b)
        return_statistics['Variance'] = (a ** 2) * var_tp    
        return_statistics['Mean'] = (a * mean_tp) + b

        


    if (measure.upper() in ['THREAT SCORE', 'CRITICAL SUCCES INDEX', 'TS', 'CSI']):
        def pmf_Y(y):
            TP_rv = hypergeom(M = M, n = P, N = int(theta * M))
            return(TP_rv.pmf((y * (P + rounded_m_theta)) / (1 + y)))
        return_statistics['Distribution'] = pmf_Y
        # return_statistics['Variance'] = (a ** 2) * var_tp    
        # return_statistics['Mean'] = (a * mean_tp) + b

        


    if (measure.upper() in ['PREVALENCE THRESHOLD', 'PT']):
        def pmf_Y(y):
            TP_rv = hypergeom(M = M, n = P, N = int(theta * M))
            return(TP_rv.pmf((rounded_m_theta * P * ((y - 1) ** 2)) / (M * (y ** 2) - 2 * P * y + P)))
        return_statistics['Distribution'] = pmf_Y
        # return_statistics['Variance'] = (a ** 2) * var_tp    
        # return_statistics['Mean'] = (a * mean_tp) + b
    
    return(return_statistics)
        
# %%
measure = 'TP'
theta = sum(true_labels) / len(true_labels)
result = basic_baseline_statistics(theta = theta, measure = measure, true_labels = true_labels)
print(result)
distribution_result = result['Distribution']
# %%
import scipy.integrate as integrate
# %%
sum([(i**2) * distribution_result(i) for i in range(0, P + 2)])
# %%

def xfx(x):
    return np.multiply(distribution_result(int(x)), int(x))

integrate.quad(xfx, a = -np.infty, b = np.infty)
# %%
