#%%
import random
predicted_labels = random.choices((0,1), k = 10000, weights = (0.99, 0.1))
true_labels = random.choices((0,1), k = 10000, weights = (0.99, 0.1))
theta = 0.5
measure = 'PT'

 
# %%
import statistics
import math
from scipy.stats import hypergeom
import numpy as np



# %%

name_dictionary = {
    'TP' : ['TP'],
    'TN' : ['TN'],
    'FP' : ['FP'],
    'FN' : ['FN'],
    'TPR' : ['TPR'],
    'TNR' : ['TNR'],
    'FPR' : ['FPR'],
    'FNR' : ['FNR'],
    'PPV' : ['PPV'],
    'NPV' : ['NPV'],
    'FDR' : ['FDR'],
    'FOR' : ['FOR'],
    'ACC' : ['ACC', 'ACCURACY'],
    'BACC' : ['BACC', 'BALANCED ACCURACY'],
    'FBETA' : ['FBETA', 'FSCORE', 'F', 'F BETA', 'F BETA SCORE', 'FBETA SCORE'],
    'MCC' : ['MCC', 'MATTHEW', 'MATTHEWS CORRELATION COEFFICIENT'],
    'BM' : ['BM', 'BOOKMAKER INFORMEDNESS', 'INFORMEDNESS'],
    'MK' : ['MARKEDNESS', 'MK'],
    'COHEN' : ['COHEN', 'COHENS KAPPA', 'KAPPA'],
    'G1' : ['GMEAN1', 'G MEAN 1', 'G1'],
    'G2' : ['GMEAN2', 'G MEAN 2', 'G2'],
    'G2 APPROX' : ['GMEAN2 APPROX', 'G MEAN 2 APPROX', 'G2 APPROX'],
    'FOWLKES': ['FOWLKES-MALLOWS', 'FOWLKES MALLOWS', 'FOWLKES', 'MALLOWS'],
    'TS' : ['THREAT SCORE', 'CRITICAL SUCCES INDEX', 'TS', 'CSI'],
    'PT' : ['PREVALENCE THRESHOLD', 'PT']
}


def select_names(name_keys):
    return(sum([name_dictionary[key_name] for key_name in name_keys] , [])) 
    
def all_names_except(name_keys):
    return(sum([name_dictionary[key_name] for key_name in name_dictionary.keys() if key_name not in name_keys] , [])) 

possible_names =  all_names_except([""])

# %%
# TODO: kan het iets sneller maken door P.predicted = TP + FP en N.predicted = TN + FN te gebruiken
def measure_score(true_labels, predicted_labels, measure = all_names_except(['G2 APPROX']), beta = 1):
    if measure not in all_names_except(['G2 APPROX']):
        raise ValueError("This measure name is not recognized.")

    if measure is possible_names:
        raise ValueError("Input a measure name.")

    if not np.unique(np.array(true_labels)) in np.array([0,1]):
        raise ValueError("true_labels should only contain zeros and ones.")

    if not np.unique(np.array(predicted_labels)) in np.array([0,1]):
        raise ValueError("predicted_labels should only contain zeros and ones.")

    P = sum(true_labels)
    M = len(true_labels)
    N = M - P    
    P.predicted = sum(predicted_labels)
    #N.predicted = M - P.predicted

    TP = np.dot(true_labels, predicted_labels)
    FP = P.predicted - TP
    FN = P - TP
    TN = N - FP

    if measure in name_dictionary('TP'):
        return(TP)

    if measure in name_dictionary('TN'):
        return(TN)
    
    if measure in name_dictionary('F'):
        return(FP)
    
    if measure in name_dictionary('FN'):
        return(FN)
    
    if measure in name_dictionary('TPR'):
        return(TP / P)

    if measure in name_dictionary('TNR'):
        return(TN / N)

    if measure in name_dictionary('FPR'):
        return(FP / N)

    if measure in name_dictionary('FNR'):
        return(FN / P)        

    if measure in name_dictionary('PPV'):
        return(TP / (TP + FP))    

    if measure in name_dictionary('NPV'):
        return(TN / (TN + FN))   

    if measure in name_dictionary('FDR'):
        return(FP / (TP + FP))    

    if measure in name_dictionary('FOR'):
        return(FN / (TN + FN))  

    if measure in name_dictionary('ACC'):
        return((TP + TN) / M)  

    if measure in name_dictionary('BACC'):
        TPR = TP / P
        TNR = TN / N
        return((TPR + TNR) / 2) 

    if measure in name_dictionary('FBETA'):
        beta_squared = beta ** 2
        return((1 + beta_squared) * TP / (((1 + beta_squared) * TP) + (beta_squared * FN) + FP))      

    if measure in name_dictionary('MCC'):
        return((TP * TN - FP * FN)/(math.sqrt((TP + FP) * (TN + FN) * P * N)))

    if measure in name_dictionary('BM'):
        TPR = TP / P
        TNR = TN / N
        return(TPR + TNR - 1)

    if measure in name_dictionary('MK'):
        PPV = TP / (TP + FP)
        NPV = TN / (TN + FN)
        return(PPV + NPV - 1)

    if measure in name_dictionary('COHEN'):
        P_o = (TP + TN) / M
        P_yes = ((TP + FP) / M) * (P / M)
        P_no = ((TN + FN) / M) * (N / M)
        P_e = P_yes + P_no
        return((P_o - P_e) / (1 - P_e))

    if measure in name_dictionary('G1'): 
        TPR = TP / P
        PPV = TP / (TP + FP)
        return(math.sqrt(TPR * PPV))   

    if measure in name_dictionary('G2'): 
        TPR = TP / P
        TNR = TN / N
        return(math.sqrt(TPR * TNR))   

    if measure in name_dictionary('FOWLKES'): 
        TPR = TP / P
        PPV = TP / (TP + FP)
        return(math.sqrt(TPR * PPV))   

    if measure in name_dictionary('PT'): 
        TPR = TP / P
        FPR = FP / N
        return((math.sqrt(TPR * FPR) - FPR) / (TPR - FPR))      

    if measure in name_dictionary('TS'):
        return(TP / (TP + FN + FP)) 

# %%


def optimized_basic_baseline(true_labels, measure = possible_names, beta = 1):

    if measure not in possible_names:
        raise ValueError("This measure name is not recognized.")
    
    if not np.array_equal(np.unique(np.array(true_labels)), np.array([0,1])):
        raise ValueError("true_labels should only contain zeros and ones with at least one of each.")

    P = sum(true_labels)
    M = len(true_labels)
    N = M - P
    return_statistics = {}
    if (measure.upper() in name_dictionary['TP']):
        return_statistics['Max Expected Value'] = P
        return_statistics['Argmax Expected Value'] = 1
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = 0

    if (measure.upper() in name_dictionary['TN']):
        return_statistics['Max Expected Value'] = N
        return_statistics['Argmax Expected Value'] = 0
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = 1

    if (measure.upper() in name_dictionary['FP']):
        return_statistics['Max Expected Value'] = N
        return_statistics['Argmax Expected Value'] = 1
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = 0

    if (measure.upper() in name_dictionary['FN']):
        return_statistics['Max Expected Value'] = P
        return_statistics['Argmax Expected Value'] = 0
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = 1

    if (measure.upper() in name_dictionary['TPR']):
        return_statistics['Max Expected Value'] = 1
        return_statistics['Argmax Expected Value'] = 1
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = 0

    if (measure.upper() in name_dictionary['TNR']):
        return_statistics['Max Expected Value'] = 1
        return_statistics['Argmax Expected Value'] = 0
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = 1

    if (measure.upper() in name_dictionary['FPR']):
        return_statistics['Max Expected Value'] = 1
        return_statistics['Argmax Expected Value'] = 1
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = 0

    if (measure.upper() in name_dictionary['FNR']):
        return_statistics['Max Expected Value'] = 1
        return_statistics['Argmax Expected Value'] = 0
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = 1

    if (measure.upper() in name_dictionary['PPV']):
        return_statistics['Max Expected Value'] = P/M
        return_statistics['Argmax Expected Value'] = [i/M for i in range(1, M + 1)]
        return_statistics['Min Expected Value'] = P/M
        return_statistics['Argmin Expected Value'] = [i/M for i in range(1, M + 1)]

    if (measure.upper() in name_dictionary['NPV']):
        return_statistics['Max Expected Value'] = N/M
        return_statistics['Argmax Expected Value'] = [i/M for i in range(0, M)]
        return_statistics['Min Expected Value'] = N/M
        return_statistics['Argmin Expected Value'] = [i/M for i in range(0, M)]

    if (measure.upper() in name_dictionary['FDR']):
        return_statistics['Max Expected Value'] = N/M
        return_statistics['Argmax Expected Value'] = [i/M for i in range(1, M + 1)]
        return_statistics['Min Expected Value'] = N/M
        return_statistics['Argmin Expected Value'] = [i/M for i in range(1, M + 1)]

    if (measure.upper() in name_dictionary['FOR']):
        return_statistics['Max Expected Value'] = P/M
        return_statistics['Argmax Expected Value'] = [i/M for i in range(0, M)]
        return_statistics['Min Expected Value'] = P/M
        return_statistics['Argmin Expected Value'] = [i/M for i in range(0, M)]

    if (measure.upper() in name_dictionary['ACC']):
        return_statistics['Max Expected Value'] = max((N/M, P/M))
        return_statistics['Argmax Expected Value'] = int((P >= N))
        return_statistics['Min Expected Value'] = min((N/M, P/M))
        return_statistics['Argmin Expected Value'] = int((P < N))

    if (measure.upper() in name_dictionary['BACC']):
        return_statistics['Max Expected Value'] = max((N/M, P/M))
        return_statistics['Argmax Expected Value'] = int((P >= N))
        return_statistics['Min Expected Value'] = min((N/M, P/M))
        return_statistics['Argmin Expected Value'] = int((P < N))

    if (measure.upper() in name_dictionary['FBETA']):
        beta_squared = beta ** 2
        return_statistics['Max Expected Value'] = (1 + beta_squared) * P / (beta_squared * P + M)
        return_statistics['Argmax Expected Value'] = 1
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = 0

    if (measure.upper() in name_dictionary['MCC']):
        return_statistics['Max Expected Value'] = 0
        return_statistics['Argmax Expected Value'] = [i/M for i in range(1, M)]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [i/M for i in range(1, M)]

    if (measure.upper() in name_dictionary['BM']):
        return_statistics['Max Expected Value'] = 0
        return_statistics['Argmax Expected Value'] = [i/M for i in range(0, M + 1)]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [i/M for i in range(0, M + 1)]

    if (measure.upper() in name_dictionary['MK']):
        return_statistics['Max Expected Value'] = 0
        return_statistics['Argmax Expected Value'] = [i/M for i in range(1, M)]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [i/M for i in range(1, M)]

    if (measure.upper() in name_dictionary['COHEN']):
        return_statistics['Max Expected Value'] = 0
        return_statistics['Argmax Expected Value'] = [i/M for i in range(0, M + 1)]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [i/M for i in range(0, M + 1)]

    if (measure.upper() in name_dictionary['G1']):
        return_statistics['Max Expected Value'] = math.sqrt(P / M)
        return_statistics['Argmax Expected Value'] = 1
        return_statistics['Min Expected Value'] = math.sqrt(P) / M
        return_statistics['Argmin Expected Value'] = 1/M

    if (measure.upper() in name_dictionary['G2']):
        print("This baseline has to be calculated. This could take some time for large M. For an approximation: use GMEAN2 APPROX")
        result = [np.nan] * (M + 1)
        for i in range(0, M + 1):
            theta = i / M
            rounded_m_theta = round(round(M * theta))
            TP_rv = hypergeom(M = M, n = P, N = rounded_m_theta)
            result[i] = sum([(math.sqrt(k * (N - rounded_m_theta + k)/ (P * N))) * TP_rv.pmf(k) if TP_rv.pmf(k) > 0 else 0 for k in range(0, min((P + 1, rounded_m_theta + 1)))])
        return_statistics['Max Expected Value'] = np.nanmax(result)
        return_statistics['Argmax Expected Value'] = [i/M for i, j in enumerate(result) if j == return_statistics['Max Expected Value'] ]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [0, 1]


    if (measure.upper() in name_dictionary['G2 APPROX']):
        print("Note: the max and argmax are an approximation.")
        return_statistics['Max Expected Value'] = 1/2
        return_statistics['Argmax Expected Value'] = 1/2
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [0, 1]



    if (measure.upper() in name_dictionary['FOWLKES']):
        return_statistics['Max Expected Value'] = math.sqrt(P / M)
        return_statistics['Argmax Expected Value'] = 1
        return_statistics['Min Expected Value'] = math.sqrt(P) / M
        return_statistics['Argmin Expected Value'] = 1/M


    if (measure.upper() in name_dictionary['TS']):
        return_statistics['Max Expected Value'] = P / M
        return_statistics['Argmax Expected Value'] = 1
        return_statistics['Min Expected Value'] = P / (P * (M - 1) + M)
        return_statistics['Argmin Expected Value'] = 1 / M


    if (measure.upper() in name_dictionary['PT']):
        result = [np.nan] * (M + 1)
        for i in [1, M - 1]:
            theta = i / M
            rounded_m_theta = round(M * theta)
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
def round_if_close(x):
    if math.isclose(x, round(x), abs_tol = 0.000001):
        return(round(x))
    else: 
        return(x)


def basic_baseline_statistics(theta, true_labels, measure = possible_names, beta = 1):
    
    if measure not in possible_names:
        raise ValueError("This measure name is not recognized.")
    
    if not np.array_equal(np.unique(np.array(true_labels)), np.array([0,1])):
        raise ValueError("true_labels should only contain zeros and ones with at least one of each.")

    # TODO: check if other measures also don't deal with theta = 1
    if theta == 1 and measure in select_names(('NPV', 'FOR', 'MCC', 'MK')):
        raise ValueError('Theta cannot be 1 with this measure')

    # TODO: check if other measures also don't deal with theta = 0
    if theta == 1 and measure in select_names(('PPV', 'FDR', 'MCC', 'MK')):
        raise ValueError('Theta cannot be 0 with this measure')

    if theta > 1 or theta < 0:
        raise ValueError('Theta must be in the interval [0,1]')


    P = sum(true_labels)
    M = len(true_labels)
    N = M - P
    rounded_m_theta = round(theta * M)
    theta_star = rounded_m_theta / M

    var_tp = (theta_star * (1 - theta_star) * P * N) / (M - 1)
    mean_tp = theta_star * P

    return_statistics = {}

    def generate_hypergeometric_distribution(a,b):
        def pmf_Y(y, theta = theta_star):
            TP_rv = hypergeom(M = M, n = P, N = round(theta * M))
            # Ik heb round toegevoegd, omdat er kleine afrond foutjes worden gemaakt
            return(TP_rv.pmf(round_if_close((y - b) / a)))
        return(pmf_Y)


    def generate_variance_function(a, b, include_0 = True, include_1 = True):
        def variance_function(theta = theta_star):
            if (theta > 1 or theta < 0) or (theta == 0 and not include_0) or (theta == 1 and not include_1):
                raise ValueError('Theta must be in the interval ' + include_0 * '[' + (not include_0) * '(' + '0,1' + include_1 * ']' + (not include_1) * ')')  
            theta_star = round(theta * M) / M
            var_tp = (theta_star * (1 - theta_star) * P * N) / (M - 1)
            return((a ** 2) * var_tp)
        return(variance_function)


    if (measure.upper() in name_dictionary['TP']):
        a = 1
        b = 0
        def expectation_function(theta = theta_star):
            if theta > 1 or theta < 0:
                raise ValueError('Theta must be in the interval [0,1]')
            theta_star = round(theta * M) / M
            return(theta_star * P)

        return_statistics['Distribution'] = generate_hypergeometric_distribution(a,b)
        return_statistics['Variance'] = (a ** 2) * var_tp
        return_statistics['Mean'] = (a * mean_tp) + b
        return_statistics['Domain'] = [(a * x) + b for x in range(0, P + 1)]
        return_statistics['Fast Expectation Function'] = expectation_function
        return_statistics['Variance Function'] = generate_variance_function(a,b)


    if (measure.upper() in name_dictionary['TN']):
        a = 1
        b = N - rounded_m_theta       
        def expectation_function(theta = theta_star):
            if theta > 1 or theta < 0:
                raise ValueError('Theta must be in the interval [0,1]')
            theta_star = round(theta * M) / M
            return((1 - theta_star) * N) 



    if (measure.upper() in name_dictionary['FP']):
        a = -1
        b = rounded_m_theta   
        def expectation_function(theta = theta_star):
            if theta > 1 or theta < 0:
                raise ValueError('Theta must be in the interval [0,1]')
            theta_star = round(theta * M) / M
            return(theta_star * N)      

    if (measure.upper() in name_dictionary['FN']):
        a = -1
        b = P
        def expectation_function(theta = theta_star):
            if theta > 1 or theta < 0:
                raise ValueError('Theta must be in the interval [0,1]')
            theta_star = round(theta * M) / M
            return((1 - theta_star) * P)   



    if (measure.upper() in name_dictionary['TPR']):
        a = 1 / P
        b = 0  
        def expectation_function(theta = theta_star):
            if theta > 1 or theta < 0:
                raise ValueError('Theta must be in the interval [0,1]')
            theta_star = round(theta * M) / M
            return(theta_star)       

        

    if (measure.upper() in name_dictionary['TNR']):
        a = 1 / N
        b = (N - rounded_m_theta) / N       
        def expectation_function(theta = theta_star):
            if theta > 1 or theta < 0:
                raise ValueError('Theta must be in the interval [0,1]')
            theta_star = round(theta * M) / M
            return(1 - theta_star) 

        
    if (measure.upper() in name_dictionary['FPR']):
        a = -1 / N
        b = rounded_m_theta / N    
        def expectation_function(theta = theta_star):
            if theta > 1 or theta < 0:
                raise ValueError('Theta must be in the interval [0,1]')
            theta_star = round(theta * M) / M
            return(theta_star)    

        
    if (measure.upper() in name_dictionary['FNR']):
        a = -1 / P
        b = 1
        def expectation_function(theta = theta_star):
            if theta > 1 or theta < 0:
                raise ValueError('Theta must be in the interval [0,1]')
            theta_star = round(theta * M) / M
            return(1 - theta_star)    


        
    if (measure.upper() in name_dictionary['PPV']):
        a = 1 / rounded_m_theta
        b = 0
        def expectation_function(theta = theta_star):
            if theta > 1 or theta < 0:
                raise ValueError('Theta must be in the interval [0,1]')
            return(P / M) 


        
        

    if (measure.upper() in name_dictionary['NPV']):
        a = 1 / (M - rounded_m_theta)
        b = (N - rounded_m_theta) / (M - rounded_m_theta)
        def expectation_function(theta = theta_star):
            if theta > 1 or theta < 0:
                raise ValueError('Theta must be in the interval [0,1]')
            return(N / M) 
        
    

    if (measure.upper() in name_dictionary['FDR']):
        a =  -1 / rounded_m_theta
        b = 1
        def expectation_function(theta = theta_star):
            if theta > 1 or theta < 0:
                raise ValueError('Theta must be in the interval [0,1]')
            return(N / M) 
        
        

    if (measure.upper() in name_dictionary['FOR']):
        a = -1 / (M - rounded_m_theta)
        b = 1 -((N - rounded_m_theta) / (M - rounded_m_theta))
        def expectation_function(theta = theta_star):
            if theta > 1 or theta < 0:
                raise ValueError('Theta must be in the interval [0,1]')
            return(P / M) 
        
        

    if (measure.upper() in name_dictionary['ACC']):
        a = 2 / M
        b = (N - rounded_m_theta) / M
        def expectation_function(theta = theta_star):
            if theta > 1 or theta < 0:
                raise ValueError('Theta must be in the interval [0,1]')
            theta_star = round(theta * M) / M
            return(((1 - theta_star) * N + (theta_star * P)) / M) 
        
        

    if (measure.upper() in name_dictionary['BACC']):
        a = (1 / (2 * P)) + (1 / (2 * N))
        b = (N - rounded_m_theta) / (2 * N)
        def expectation_function(theta = theta_star):
            if theta > 1 or theta < 0:
                raise ValueError('Theta must be in the interval [0,1]')
            return(1 / 2) 

        
        

    if (measure.upper() in name_dictionary['FBETA']):
        beta_squared = beta ** 2
        a = (1 + beta_squared) / (beta_squared * P + M * theta_star)
        b = 0
        def expectation_function(theta = theta_star, beta = beta):
            if theta > 1 or theta < 0:
                raise ValueError('Theta must be in the interval [0,1]')            
            theta_star = round(theta * M) / M
            beta_squared = beta ** 2
            return(((1 + beta_squared) * theta_star * P) / (beta_squared * P + M * theta_star)) 

        
    

    if (measure.upper() in name_dictionary['MCC']):
        a = 1 / (math.sqrt(theta_star * (1 - theta_star) * P * N))
        b = - theta_star * P / (math.sqrt(theta_star * (1 - theta_star) * P * N))
        def expectation_function(theta = theta_star):
            if theta > 1 or theta < 0:
                raise ValueError('Theta must be in the interval [0,1]')
            return(0) 

        

    if (measure.upper() in name_dictionary['BM']):
        a = (1 / P) + (1 / N)
        b = - rounded_m_theta / N
        def expectation_function(theta = theta_star):
            if theta > 1 or theta < 0:
                raise ValueError('Theta must be in the interval [0,1]')
            return(0) 

                

    if (measure.upper() in name_dictionary['MK']):
        a = (1 / rounded_m_theta) + (1 / (M - rounded_m_theta))
        b =  -P / (M - rounded_m_theta)
        def expectation_function(theta = theta_star):
            if theta > 1 or theta < 0:
                raise ValueError('Theta must be in the interval [0,1]')
            return(0) 

            

    if (measure.upper() in name_dictionary['COHEN']):
        a = 2 / ((1 - theta_star) * P + theta_star * N)
        b = - 2 * theta_star * P / ((1 - theta_star) * P + theta_star * N)
        def expectation_function(theta = theta_star):
            if theta > 1 or theta < 0:
                raise ValueError('Theta must be in the interval [0,1]')
            return(0) 
                

    if (measure.upper() in name_dictionary['G1']):
        a = 1 / (math.sqrt(P * rounded_m_theta))
        b = 0
        def expectation_function(theta = theta_star):
            if theta > 1 or theta < 0:
                raise ValueError('Theta must be in the interval [0,1]')
            theta_star = round(theta * M) / M
            return(math.sqrt(theta_star * P / M)) 

                

    if (measure.upper() in name_dictionary['G2']):
        def pmf_Y(y, theta = theta_star):
            TP_rv = hypergeom(M = M, n = P, N = round(theta * M))
            rounded_m_theta = round(theta * M)
            help_constant = math.sqrt((rounded_m_theta ** 2) - 2 * rounded_m_theta * N + (N ** 2) + 4 * P * N * (y ** 2))
            value_1 = (1/2) * ((- help_constant) + rounded_m_theta - N)
            value_2 = (1/2) * (help_constant + rounded_m_theta - N)
            return(TP_rv.pmf(round_if_close(value_1)) + TP_rv.pmf(round_if_close(value_2)))
        
        def G_mean_2_given_tp(x, theta = theta_star):
            rounded_m_theta = round(theta * M) 
            return(math.sqrt((x / P) * ((N - rounded_m_theta + x) / N)))


        def expectation_function(theta = theta_star):
            if theta > 1 or theta < 0:
                raise ValueError('Theta must be in the interval [0,1]')
            TP_rv = hypergeom(M = M, n = P, N = round(theta * M))
            return(sum([TP_rv.pmf(x) * G_mean_2_given_tp(x, theta) for x in range(0, P + 1)]))


        def variance_function(theta = theta_star):
            if theta > 1 or theta < 0:
                raise ValueError('Theta must be in the interval [0,1]')
            TP_rv = hypergeom(M = M, n = P, N = round(theta * M))
            return(sum([TP_rv.pmf(x) * (G_mean_2_given_tp(x, theta) ** 2) for x in range(0, P + 1)]))

        TP_rv = hypergeom(M = M, n = P, N = round(theta * M))

        return_statistics['Distribution'] = pmf_Y
        return_statistics['Mean'] = expectation_function(theta_star)
        return_statistics['Variance'] = variance_function(theta_star)
        return_statistics['Domain'] = np.unique([G_mean_2_given_tp(x) for x in range(0, P + 1)])
        return_statistics['Expectation Function'] = expectation_function
        return_statistics['Variance Function'] = variance_function             

        

    if (measure.upper() in name_dictionary['FOWLKES']):
        a = 1 / (math.sqrt(P * rounded_m_theta))
        b = 0
        def expectation_function(theta = theta_star):
            if theta > 1 or theta < 0:
                raise ValueError('Theta must be in the interval [0,1]')
            theta_star = round(theta * M) / M
            return(math.sqrt(theta_star * P / M)) 
              

    if (measure.upper() in name_dictionary['TS']):
        def pmf_Y(y, theta = theta_star):
            TP_rv = hypergeom(M = M, n = P, N = round(theta * M))
            rounded_m_theta = round(theta * M)
            return(TP_rv.pmf(round_if_close((y * (P + rounded_m_theta)) / (1 + y))))
        
        def TS_given_tp(x, theta = theta_star):
            rounded_m_theta = round(theta * M)
            if P + rounded_m_theta - x == 0:
                return(0)
            else:
                return(x / (P + rounded_m_theta - x))

        def expectation_function(theta = theta_star):
            if theta > 1 or theta < 0:
                raise ValueError('Theta must be in the interval [0,1]')
            TP_rv = hypergeom(M = M, n = P, N = round(theta * M))
            return(sum([TP_rv.pmf(x) * TS_given_tp(x, theta) for x in range(0, P + 1)]))

        def variance_function(theta = theta_star):
            if theta > 1 or theta < 0:
                raise ValueError('Theta must be in the interval [0,1]')
            TP_rv = hypergeom(M = M, n = P, N = round(theta * M))
            return(sum([TP_rv.pmf(x) * (TS_given_tp(x, theta) ** 2) for x in range(0, P + 1)]))

        TP_rv = hypergeom(M = M, n = P, N = round(theta * M))

        return_statistics['Distribution'] = pmf_Y
        return_statistics['Mean'] = expectation_function(theta_star)
        return_statistics['Variance'] = variance_function(theta_star)
        return_statistics['Domain'] = np.unique([TS_given_tp(x) for x in range(0, P + 1)])
        return_statistics['Expectation Function'] = expectation_function
        return_statistics['Variance Function'] = variance_function        

    if (measure.upper() in name_dictionary['PT']):
        def pmf_Y(y, theta = theta_star):
            TP_rv = hypergeom(M = M, n = P, N = round(theta * M))
            rounded_m_theta = round(theta * M)            
            return(TP_rv.pmf(round_if_close((rounded_m_theta * P * ((y - 1) ** 2)) / (M * (y ** 2) - 2 * P * y + P))))

        def PT_given_tp(x, theta = theta_star):
            rounded_m_theta = round(theta * M)
            help_1 = x / P
            help_2 = (x - rounded_m_theta) / N
            if help_1 + help_2 == 0:
                return(0)
            else: 
                return((math.sqrt(help_1 * (- help_2)) + help_2)   / (help_1 + help_2))        
        
        def expectation_function(theta = theta_star):
            if theta > 1 or theta < 0:
                raise ValueError('Theta must be in the interval [0,1]')
            TP_rv = hypergeom(M = M, n = P, N = round(theta * M))
            return(sum([TP_rv.pmf(x) * PT_given_tp(x, theta) for x in range(0, P + 1)]))

        def variance_function(theta = theta_star):
            if theta > 1 or theta < 0:
                raise ValueError('Theta must be in the interval [0,1]')
            TP_rv = hypergeom(M = M, n = P, N = round(theta * M))
            return(sum([TP_rv.pmf(x) * (PT_given_tp(x, theta) ** 2) for x in range(0, P + 1)]))

        TP_rv = hypergeom(M = M, n = P, N = round(theta * M))

        return_statistics['Distribution'] = pmf_Y
        return_statistics['Mean'] = expectation_function(theta_star)
        return_statistics['Variance'] = variance_function(theta_star)
        return_statistics['Domain'] = np.unique([PT_given_tp(x) for x in range(0, P + 1)])
        return_statistics['Expectation Function'] = expectation_function
        return_statistics['Variance Function'] = variance_function        


    if (measure.upper() in all_names_except(['G2', 'G2 APPROX', 'TS', 'PT'])):
        return_statistics['Distribution'] = generate_hypergeometric_distribution(a,b)
        return_statistics['Variance'] = (a ** 2) * var_tp
        return_statistics['Mean'] = (a * mean_tp) + b
        return_statistics['Domain'] = [(a * x) + b for x in range(0, P + 1)]
        return_statistics['Fast Expectation Function'] = expectation_function
        return_statistics['Variance Function'] = generate_variance_function(a,b)



    return(return_statistics)
        




# %%


measures_list = [i for i in name_dictionary.keys()]

theta = 1/4
mean_list = []
for i, measure in enumerate(measures_list):
    result = basic_baseline_statistics(theta = theta, measure = measure, true_labels = true_labels)
    print(measure + ":" + str(result['Mean']))
    mean_list.append(result['Mean'])


# %%

# %%

# %%
