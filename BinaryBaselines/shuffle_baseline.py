# %%
import random
import statistics
import math
from scipy.stats import hypergeom
import numpy as np
from functools import wraps

__all__ = ['all_names_except', 'basic_baseline_given_theta', 'measure_score', 'name_dictionary',
           'optimized_basic_baseline', 'round_if_close', 'select_names']

# %%

name_dictionary = {
    'TP': ['TP'],
    'TN': ['TN'],
    'FP': ['FP'],
    'FN': ['FN'],
    'TPR': ['TPR'],
    'TNR': ['TNR'],
    'FPR': ['FPR'],
    'FNR': ['FNR'],
    'PPV': ['PPV'],
    'NPV': ['NPV'],
    'FDR': ['FDR'],
    'FOR': ['FOR'],
    'ACC': ['ACC', 'ACCURACY'],
    'BACC': ['BACC', 'BALANCED ACCURACY'],
    'FBETA': ['FBETA', 'FSCORE', 'F', 'F BETA', 'F BETA SCORE', 'FBETA SCORE'],
    'MCC': ['MCC', 'MATTHEW', 'MATTHEWS CORRELATION COEFFICIENT'],
    'BM': ['BM', 'BOOKMAKER INFORMEDNESS', 'INFORMEDNESS'],
    'MK': ['MARKEDNESS', 'MK'],
    'COHEN': ['COHEN', 'COHENS KAPPA', 'KAPPA'],
    'G1': ['GMEAN1', 'G MEAN 1', 'G1'],
    'G2': ['GMEAN2', 'G MEAN 2', 'G2', 'FOWLKES-MALLOWS', 'FOWLKES MALLOWS', 'FOWLKES', 'MALLOWS'],
    'TS': ['THREAT SCORE', 'CRITICAL SUCCES INDEX', 'TS', 'CSI'],
    'PT': ['PREVALENCE THRESHOLD', 'PT']
}


def select_names(name_keys):
    """
    This function creates a list of names using the name_keys as keys for the name dictionary.
    """
    return(sum([name_dictionary[key_name] for key_name in name_keys], []))


def all_names_except(name_keys):
    """
    This function creates a list of all names, except the names with name_keys as key in the name dictionary.
    """
    return(sum([name_dictionary[key_name] for key_name in name_dictionary.keys() if key_name not in name_keys], []))


def measure_score(true_labels, predicted_labels, measure, beta=1):
    """
    To determine the performance of a predictive model a measure is used. This function determines the measure for the given input labels.

    Args:
    --------
        true_labels (list): 1-dimensional boolean list containing the true labels.

        predicted_labels (list): 1-dimensional boolean list containing the predicted labels.

        measure (string): Measure name, see `all_names_except([''])` for possible measure names.

        beta (float): Default is 1. Parameter for the F-beta score.

    Returns:
    --------
        float: The score of the given measure evaluated with the predicted and true labels.

    Raises:
    --------
        ValueError 
            If `measure` is not in `all_names_except([''])`.
        ValueError
            If `true_labels` or `predicted_labels` does not only contain zeros and ones.
        TypeError
            If `true_labels` or `predicted_labels` is not a list.

    See also:
    --------
        all_names_except

    Example:
    --------
        >>> import random
        >>> random.seed(123) # To ensure similar outputs
        >>> predicted_labels = random.choices((0,1), k = 10000, weights = (0.9, 0.1))
        >>> true_labels = random.choices((0,1), k = 10000, weights = (0.9, 0.1))
        >>> print('Markedness: {:06.4f}'.format(measure_score(true_labels, predicted_labels, measure = 'MK'))) # Measuring markedness (MK)
        Markedness: 0.0061
        >>> print('F2 Score: {:06.4f}'.format(measure_score(true_labels, predicted_labels, measure = 'FBETA', beta = 2))) # Measuring FBETA for beta = 2
        F2 Score: 0.1053

    """
    measure = measure.upper()

    if measure not in all_names_except(['']):
        raise ValueError("This measure name is not recognized.")

    if type(true_labels) != list:
        raise TypeError('true_labels should be a list')

    if type(predicted_labels) != list:
        raise TypeError('predicted_labels should be a list')

    if not np.unique(np.array(true_labels)) in np.array([0, 1]):
        raise ValueError("true_labels should only contain zeros and ones.")

    if not np.unique(np.array(predicted_labels)) in np.array([0, 1]):
        raise ValueError(
            "predicted_labels should only contain zeros and ones.")

    P = sum(true_labels)
    M = len(true_labels)
    N = M - P
    P_predicted = sum(predicted_labels)

    TP = np.dot(true_labels, predicted_labels)
    FP = P_predicted - TP
    FN = P - TP
    TN = N - FP

    if measure in name_dictionary['TP']:
        return(TP)

    if measure in name_dictionary['TN']:
        return(TN)

    if measure in name_dictionary['FP']:
        return(FP)

    if measure in name_dictionary['FN']:
        return(FN)

    if measure in name_dictionary['TPR']:
        return(TP / P)

    if measure in name_dictionary['TNR']:
        return(TN / N)

    if measure in name_dictionary['FPR']:
        return(FP / N)

    if measure in name_dictionary['FNR']:
        return(FN / P)

    if measure in name_dictionary['PPV']:
        return(TP / (TP + FP))

    if measure in name_dictionary['NPV']:
        return(TN / (TN + FN))

    if measure in name_dictionary['FDR']:
        return(FP / (TP + FP))

    if measure in name_dictionary['FOR']:
        return(FN / (TN + FN))

    if measure in name_dictionary['ACC']:
        return((TP + TN) / M)

    if measure in name_dictionary['BACC']:
        TPR = TP / P
        TNR = TN / N
        return((TPR + TNR) / 2)

    if measure in name_dictionary['FBETA']:
        beta_squared = beta ** 2
        return((1 + beta_squared) * TP / (((1 + beta_squared) * TP) + (beta_squared * FN) + FP))

    if measure in name_dictionary['MCC']:
        return((TP * TN - FP * FN)/(math.sqrt((TP + FP) * (TN + FN) * P * N)))

    if measure in name_dictionary['BM']:
        TPR = TP / P
        TNR = TN / N
        return(TPR + TNR - 1)

    if measure in name_dictionary['MK']:
        PPV = TP / (TP + FP)
        NPV = TN / (TN + FN)
        return(PPV + NPV - 1)

    if measure in name_dictionary['COHEN']:
        P_o = (TP + TN) / M
        P_yes = ((TP + FP) / M) * (P / M)
        P_no = ((TN + FN) / M) * (N / M)
        P_e = P_yes + P_no
        return((P_o - P_e) / (1 - P_e))

    if measure in name_dictionary['G1']:
        TPR = TP / P
        PPV = TP / (TP + FP)
        return(math.sqrt(TPR * PPV))

    if measure in name_dictionary['G2']:
        TPR = TP / P
        TNR = TN / N
        return(math.sqrt(TPR * TNR))

    if measure in name_dictionary['PT']:
        TPR = TP / P
        FPR = FP / N
        return((math.sqrt(TPR * FPR) - FPR) / (TPR - FPR))

    if measure in name_dictionary['TS']:
        return(TP / (TP + FN + FP))


def optimized_basic_baseline(true_labels, measure, beta=1):
    """
    This function determines the optimal `theta` that maximizes or minimizes the measure on the `true_labels`. It also determines the corresponding extreme value.

    Args:
    --------
        true_labels (list): 1-dimensional boolean list containing the true labels.

        measure (string): Measure name, see `all_names_except([''])` for possible measure names.

        beta (float): Default is 1. Parameter for the F-beta score.

    Returns:
    --------
        dict: Containing `Max Expected Value`, `Argmax Expected Value`, `Min Expected Value` and `Argmin Expected Value`. 

            - `Max Expected Value` (float): Maximum of the expected values for all `theta`.

            - `Argmax Expected Value` (list): List of all `theta` values that maximize the expected value.

            - `Min Expected Value` (float): Minimum of the expected values for all `theta`.

            - `Argmin Expected Value` (list): List of all `theta` values that minimize the expected value.

    Raises:
    --------
        ValueError 
            If `measure` is not in `all_names_except([''])`.
        ValueError
            If `true_labels` does not only contain zeros and ones.
        TypeError
            If `true_labels` is not a list.

    See also:
    --------
        all_names_except
        basic_baseline


    Example:
    --------
        >>> import random
        >>> random.seed(123) # To ensure similar outputs
        >>> true_labels = random.choices((0,1), k = 10000, weights = (0.9, 0.1))
        >>> optimal_baseline = optimized_basic_baseline(true_labels, measure = 'FBETA', beta = 1)
        >>> print('Max Expected Value: {:06.4f}'.format(optimal_baseline['Max Expected Value']))
        Max Expected Value: 0.1805
        >>> print('Argmax Expected Value: {:06.4f}'.format(optimal_baseline['Argmax Expected Value']))
        Argmax Expected Value: 1.0000
        >>> print('Min Expected Value: {:06.4f}'.format(optimal_baseline['Min Expected Value']))
        Min Expected Value: 0.0000
        >>> print('Argmin Expected Value: {:06.4f}'.format(optimal_baseline['Argmin Expected Value']))
        Argmin Expected Value: 0.0000
    """

    measure = measure.upper()

    if measure not in all_names_except(['']):
        raise ValueError("This measure name is not recognized.")

    if type(true_labels) != list:
        raise TypeError('true_labels should be a list')

    if not np.unique(np.array(true_labels)) in np.array([0, 1]):
        raise ValueError("true_labels should only contain zeros and ones.")

    P = sum(true_labels)
    M = len(true_labels)
    N = M - P
    return_statistics = {}

    if (measure in name_dictionary['TP']):
        return_statistics['Max Expected Value'] = P
        return_statistics['Argmax Expected Value'] = [1]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [0]

    if (measure in name_dictionary['TN']):
        return_statistics['Max Expected Value'] = N
        return_statistics['Argmax Expected Value'] = [0]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [1]

    if (measure in name_dictionary['FP']):
        return_statistics['Max Expected Value'] = N
        return_statistics['Argmax Expected Value'] = [1]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [0]

    if (measure in name_dictionary['FN']):
        return_statistics['Max Expected Value'] = P
        return_statistics['Argmax Expected Value'] = [0]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [1]

    if (measure in name_dictionary['TPR']):
        return_statistics['Max Expected Value'] = 1
        return_statistics['Argmax Expected Value'] = [1]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [0]

    if (measure in name_dictionary['TNR']):
        return_statistics['Max Expected Value'] = 1
        return_statistics['Argmax Expected Value'] = [0]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [1]

    if (measure in name_dictionary['FPR']):
        return_statistics['Max Expected Value'] = 1
        return_statistics['Argmax Expected Value'] = [1]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [0]

    if (measure in name_dictionary['FNR']):
        return_statistics['Max Expected Value'] = 1
        return_statistics['Argmax Expected Value'] = [0]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [1]

    if (measure in name_dictionary['PPV']):
        return_statistics['Max Expected Value'] = P/M
        return_statistics['Argmax Expected Value'] = [
            i/M for i in range(1, M + 1)]
        return_statistics['Min Expected Value'] = P/M
        return_statistics['Argmin Expected Value'] = [
            i/M for i in range(1, M + 1)]

    if (measure in name_dictionary['NPV']):
        return_statistics['Max Expected Value'] = N/M
        return_statistics['Argmax Expected Value'] = [i/M for i in range(0, M)]
        return_statistics['Min Expected Value'] = N/M
        return_statistics['Argmin Expected Value'] = [i/M for i in range(0, M)]

    if (measure in name_dictionary['FDR']):
        return_statistics['Max Expected Value'] = N/M
        return_statistics['Argmax Expected Value'] = [
            i/M for i in range(1, M + 1)]
        return_statistics['Min Expected Value'] = N/M
        return_statistics['Argmin Expected Value'] = [
            i/M for i in range(1, M + 1)]

    if (measure in name_dictionary['FOR']):
        return_statistics['Max Expected Value'] = P/M
        return_statistics['Argmax Expected Value'] = [i/M for i in range(0, M)]
        return_statistics['Min Expected Value'] = P/M
        return_statistics['Argmin Expected Value'] = [i/M for i in range(0, M)]

    if (measure in name_dictionary['ACC']):
        return_statistics['Max Expected Value'] = max((N/M, P/M))
        return_statistics['Argmax Expected Value'] = [int((P >= N))]
        return_statistics['Min Expected Value'] = min((N/M, P/M))
        return_statistics['Argmin Expected Value'] = [int((P < N))]

    if (measure in name_dictionary['BACC']):
        return_statistics['Max Expected Value'] = max((N/M, P/M))
        return_statistics['Argmax Expected Value'] = [int((P >= N))]
        return_statistics['Min Expected Value'] = min((N/M, P/M))
        return_statistics['Argmin Expected Value'] = [int((P < N))]

    if (measure in name_dictionary['FBETA']):
        beta_squared = beta ** 2
        return_statistics['Max Expected Value'] = (
            1 + beta_squared) * P / (beta_squared * P + M)
        return_statistics['Argmax Expected Value'] = [1]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [0]

    if (measure in name_dictionary['MCC']):
        return_statistics['Max Expected Value'] = 0
        return_statistics['Argmax Expected Value'] = [i/M for i in range(1, M)]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [i/M for i in range(1, M)]

    if (measure in name_dictionary['BM']):
        return_statistics['Max Expected Value'] = 0
        return_statistics['Argmax Expected Value'] = [
            i/M for i in range(0, M + 1)]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [
            i/M for i in range(0, M + 1)]

    if (measure in name_dictionary['MK']):
        return_statistics['Max Expected Value'] = 0
        return_statistics['Argmax Expected Value'] = [i/M for i in range(1, M)]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [i/M for i in range(1, M)]

    if (measure in name_dictionary['COHEN']):
        return_statistics['Max Expected Value'] = 0
        return_statistics['Argmax Expected Value'] = [
            i/M for i in range(0, M + 1)]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [
            i/M for i in range(0, M + 1)]

    if (measure in name_dictionary['G1']):
        return_statistics['Max Expected Value'] = math.sqrt(P / M)
        return_statistics['Argmax Expected Value'] = [1]
        return_statistics['Min Expected Value'] = math.sqrt(P) / M
        return_statistics['Argmin Expected Value'] = [1/M]

    if (measure in name_dictionary['G2']):
        result = [np.nan] * (M + 1)
        for i in range(0, M + 1):
            theta = i / M
            rounded_m_theta = round(round(M * theta))
            TP_rv = hypergeom(M=M, n=P, N=rounded_m_theta)
            result[i] = sum([(math.sqrt(k * (N - rounded_m_theta + k) / (P * N))) * TP_rv.pmf(
                k) if TP_rv.pmf(k) > 0 else 0 for k in range(0, min((P + 1, rounded_m_theta + 1)))])
        return_statistics['Max Expected Value'] = np.nanmax(result)
        return_statistics['Argmax Expected Value'] = [
            i/M for i, j in enumerate(result) if j == return_statistics['Max Expected Value']]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [0, 1]

    if (measure in name_dictionary['TS']):
        return_statistics['Max Expected Value'] = P / M
        return_statistics['Argmax Expected Value'] = [1]
        return_statistics['Min Expected Value'] = P / (P * (M - 1) + M)
        return_statistics['Argmin Expected Value'] = [1 / M]

    if (measure in name_dictionary['PT']):
        result = [np.nan] * (M + 1)
        for i in [1, M - 1]:
            theta = i / M
            rounded_m_theta = round(M * theta)
            TP_rv = hypergeom(M=M, n=P, N=rounded_m_theta)
            result[i] = sum([((math.sqrt((k / P) * (-(k - rounded_m_theta) / N)) + ((k - rounded_m_theta) / N)) / ((k / P) + ((k - rounded_m_theta) / N)))
                             * TP_rv.pmf(k) if ((k / P) + ((k - rounded_m_theta) / N)) != 0 else 0 for k in range(0, min((P + 1, rounded_m_theta + 1)))])
        return_statistics['Max Expected Value'] = np.nanmax(result)
        return_statistics['Argmax Expected Value'] = [1 / M]
        return_statistics['Min Expected Value'] = np.nanmin(result)
        return_statistics['Argmin Expected Value'] = [(M - 1) / M]

    return(return_statistics)


def round_if_close(x):
    """
    This function is used to round x if it is close. This is useful for the pmf of the hypergeometric distribution.
    """
    if math.isclose(x, round(x), abs_tol=0.000001):
        return(round(x))
    else:
        return(x)


def add_check_theta_generator(measure):
    """
    This is a decorator to add a ValueError to a function if theta is not in the proper interval.
    """
    include_0 = True
    include_1 = True
    measure = measure.upper()
    # Should 0 be included
    if measure in select_names(['PPV', 'FDR', 'MCC', 'MK', 'PT', 'G1']):
        include_0 = False
    # Should 1 be included
    if measure in select_names(['NPV', 'FOR', 'MCC', 'MK', 'PT']):
        include_1 = False

    def add_check_theta(func):
        @wraps(func)
        def inner(theta, *args, **kwargs):
            if (theta > 1 or theta < 0) or (theta == 0 and not include_0) or (theta == 1 and not include_1):
                raise ValueError('Theta must be in the interval ' + include_0 * '[' + (
                    not include_0) * '(' + '0,1' + include_1 * ']' + (not include_1) * ')')
            return func(theta, *args, **kwargs)
        return inner
    return add_check_theta


def basic_baseline(true_labels, measure, beta=1):
    """
    This function returns a dictionary of functions that can be used to determine statistics (such as expectation and variance) for all possible values of `theta`. 

    Args:
    --------
        true_labels (list): 1-dimensional boolean list containing the true labels.

        measure (string): Measure name, see `all_names_except([''])` for possible measure names.

        beta (float): Default is 1. Parameter for the F-beta score.

    Returns:
    --------
        dict: Containing `Distribution`, `Domain`, `(Fast) Expectation Function` and `Variance Function`. 

            - `Distribution` (function): Pmf of the measure, given by: `pmf_Y(y, theta)`, where `y` is a measure score and `theta` is the parameter of the shuffle baseline.

            - `Domain` (function): Function that returns attainable measure scores with argument `theta`.

            - `(Fast) Expectation Function` (function): Expectation function of the baseline with `theta` as argument. If `Fast Expectation Function` is returned, there exists a theoretical expectation that can be used for fast computation.

            - `Variance Function` (function): Variance function for all values of `theta`.

    Raises:
    --------
        ValueError 
            If `measure` is not in `all_names_except([''])`.
        ValueError
            If `true_labels` does not only contain zeros and ones.
        TypeError
            If `true_labels` is not a list.

    See also:
    --------
        all_names_except
        select_names
        round_if_close

    Example:
    --------
        >>> import random
        >>> random.seed(123) # To ensure similar outputs
        >>> true_labels = random.choices((0,1), k = 10000, weights = (0.9, 0.1))
        >>> baseline = basic_baseline(true_labels, 'MK')
        >>> print(baseline.keys())
        dict_keys(['Distribution', 'Domain', 'Fast Expectation Function', 'Variance Function', 'Expectation Function'])
    """

    measure = measure.upper()

    if measure not in all_names_except(['']):
        raise ValueError("This measure name is not recognized.")

    if type(true_labels) != list:
        raise TypeError('true_labels should be a list')

    if not np.unique(np.array(true_labels)) in np.array([0, 1]):
        raise ValueError("true_labels should only contain zeros and ones.")

    P = sum(true_labels)
    M = len(true_labels)
    N = M - P

    # Used to return all functions
    return_functions = {}

    # Used to generate pmf functions
    def generate_hypergeometric_distribution(a, b):
        def pmf_Y(y, theta):
            TP_rv = hypergeom(M=M, n=P, N=round(theta * M))
            # Use round_if_close function, because of small computation errors in python
            return(TP_rv.pmf(round_if_close((y - b) / a)))
        return(pmf_Y)

    # Used to generate variance functions
    def generate_variance_function(a):
        @add_check_theta_generator(measure)
        def variance_function(theta):
            theta_star = round(theta * M) / M
            rounded_m_theta = round(theta * M)
            var_tp = (theta_star * (1 - theta_star) * P * N) / (M - 1)
            return((eval(a) ** 2) * var_tp)
        return(variance_function)

    # Used to generate expectation functions
    def generate_expectation_function(a, b):
        @add_check_theta_generator(measure)
        def expectation_function(theta):
            theta_star = round(theta * M) / M
            rounded_m_theta = round(theta * M)
            mean_tp = theta_star * P
            return(eval(a) * mean_tp + eval(b))
        return(expectation_function)

    # Used to generate fast expectation functions. The expectation string is used to alter the function.
    def generate_fast_expectation_function(expectation_string):
        @add_check_theta_generator(measure)
        def fast_expectation_function(theta):
            theta_star = round(theta * M) / M
            return(eval(expectation_string))
        return(fast_expectation_function)

    # Used to generate domain functions
    def generate_domain_function(a, b):
        @add_check_theta_generator(measure)
        def domain_function(theta):
            theta_star = round(theta * M) / M
            rounded_m_theta = round(theta * M)
            return([(eval(a) * x) + eval(b) for x in range(0, min((P + 1, rounded_m_theta + 1)))])
        return(domain_function)

    # Used to generate domain function for PT, TS and G2.
    def generate_domain_function_given_x(given_x_function):
        @add_check_theta_generator(measure)
        def domain_function(theta):
            rounded_m_theta = round(theta * M)
            return(np.unique([given_x_function(x, theta) for x in range(0, min((P + 1, rounded_m_theta + 1)))]))
        return(domain_function)

    if (measure in name_dictionary['TP']):
        a = '1'
        b = '0'
        expectation_string = 'theta_star * ' + str(P)

    if (measure in name_dictionary['TN']):
        a = '1'
        b = str(N) + ' - rounded_m_theta'
        expectation_string = '(1 - theta_star) * ' + str(N)

    if (measure in name_dictionary['FP']):
        a = '-1'
        b = 'rounded_m_theta'
        expectation_string = 'theta_star * ' + str(N)

    if (measure in name_dictionary['FN']):
        a = '-1'
        b = str(P)
        expectation_string = '(1 - theta_star) * ' + str(P)

    if (measure in name_dictionary['TPR']):
        a = '1 / ' + str(P)
        b = '0'
        expectation_string = 'theta_star'

    if (measure in name_dictionary['TNR']):
        a = '1 / ' + str(N)
        b = '(' + str(N) + ' - rounded_m_theta) / ' + str(N)
        expectation_string = '1 - theta_star'

    if (measure in name_dictionary['FPR']):
        a = '-1 / ' + str(N)
        b = 'rounded_m_theta / ' + str(N)
        expectation_string = 'theta_star'

    if (measure in name_dictionary['FNR']):
        a = '-1 / ' + str(P)
        b = '1'
        expectation_string = '1 - theta_star'

    if (measure in name_dictionary['PPV']):
        a = '1 / rounded_m_theta'
        b = '0'
        expectation_string = str(P) + ' / ' + str(M)

    if (measure in name_dictionary['NPV']):
        a = '1 / (' + str(M) + ' - rounded_m_theta)'
        b = '(' + str(N) + ' - rounded_m_theta) / (' + \
            str(M) + ' - rounded_m_theta)'
        expectation_string = str(N) + ' / ' + str(M)

    if (measure in name_dictionary['FDR']):
        a = '-1 / rounded_m_theta'
        b = '1'
        expectation_string = str(N) + ' / ' + str(M)

    if (measure in name_dictionary['FOR']):
        a = '-1 / (' + str(M) + ' - rounded_m_theta)'
        b = '1 - ((' + str(N) + ' - rounded_m_theta) / (' + \
            str(M) + ' - rounded_m_theta))'
        expectation_string = str(P) + ' / ' + str(M)

    if (measure in name_dictionary['ACC']):
        a = '2 / ' + str(M)
        b = '(' + str(N) + ' - rounded_m_theta) / ' + str(M)
        expectation_string = '((1 - theta_star) * ' + str(N) + \
            ' + (theta_star * ' + str(P) + ')) / ' + str(M)

    if (measure in name_dictionary['BACC']):
        a = '(1 / (2 * ' + str(P) + ')) + (1 / (2 * ' + str(N) + '))'
        b = '(' + str(N) + ' - rounded_m_theta) / (2 * ' + str(N) + ')'
        expectation_string = '1 / 2'

    if (measure in name_dictionary['FBETA']):
        a = '(1 + (' + str(beta) + ' ** 2)) / ((' + str(beta) + \
            ' ** 2) * ' + str(P) + ' + ' + str(M) + ' * theta_star)'
        b = '0'
        expectation_string = '((1 + (' + str(beta) + ' ** 2)) * theta_star * ' + str(
            P) + ') / (' + str(beta) + ' ** 2) * ' + str(P) + ' + ' + str(M) + ' * theta_star)'

    if (measure in name_dictionary['MCC']):
        a = '1 / (math.sqrt(theta_star * (1 - theta_star) * ' + \
            str(P) + ' * ' + str(N) + '))'
        b = '- theta_star * ' + \
            str(P) + ' / (math.sqrt(theta_star * (1 - theta_star) * ' + \
            str(P) + ' * ' + str(N) + '))'
        expectation_string = '0'

    if (measure in name_dictionary['BM']):
        a = '(1 / ' + str(P) + ') + (1 / ' + str(N) + ')'
        b = '- rounded_m_theta / ' + str(N)
        expectation_string = '0'

    if (measure in name_dictionary['MK']):
        a = '(1 / rounded_m_theta) + (1 / (' + str(M) + ' - rounded_m_theta))'
        b = '-' + str(P) + ' / (' + str(M) + ' - rounded_m_theta)'
        expectation_string = '0'

    if (measure in name_dictionary['COHEN']):
        a = '2 / ((1 - theta_star) * ' + str(P) + \
            ' + theta_star * ' + str(N) + ')'
        b = '- 2 * theta_star * ' + \
            str(P) + ' / ((1 - theta_star) * ' + \
            str(P) + ' + theta_star * ' + str(N) + ')'
        expectation_string = '0'

    if (measure in name_dictionary['G1']):
        a = '1 / (math.sqrt(' + str(P) + ' * rounded_m_theta))'
        b = '0'
        expectation_string = 'math.sqrt(theta_star * ' + \
            str(P) + ' / ' + str(M) + ')'

    if (measure in name_dictionary['G2']):
        @add_check_theta_generator(measure)
        def pmf_Y(y, theta):
            TP_rv = hypergeom(M=M, n=P, N=round(theta * M))
            rounded_m_theta = round(theta * M)
            help_constant = math.sqrt(
                (rounded_m_theta ** 2) - 2 * rounded_m_theta * N + (N ** 2) + 4 * P * N * (y ** 2))
            value_1 = (1/2) * ((- help_constant) + rounded_m_theta - N)
            value_2 = (1/2) * (help_constant + rounded_m_theta - N)
            return(TP_rv.pmf(round_if_close(value_1)) + TP_rv.pmf(round_if_close(value_2)))

        def given_x_function(x, theta):
            rounded_m_theta = round(theta * M)
            return(math.sqrt((x / P) * ((N - rounded_m_theta + x) / N)))

        @add_check_theta_generator(measure)
        def expectation_function(theta):
            rounded_m_theta = round(theta * M)
            TP_rv = hypergeom(M=M, n=P, N=round(theta * M))
            return(sum([TP_rv.pmf(x) * given_x_function(x, theta) for x in range(0, min((P + 1, rounded_m_theta + 1)))]))

        @add_check_theta_generator(measure)
        def variance_function(theta):
            rounded_m_theta = round(theta * M)
            TP_rv = hypergeom(M=M, n=P, N=round(theta * M))
            return(sum([TP_rv.pmf(x) * (given_x_function(x, theta) ** 2) for x in range(0, min((P + 1, rounded_m_theta + 1)))]))

    if (measure in name_dictionary['TS']):
        @add_check_theta_generator(measure)
        def pmf_Y(y, theta):
            TP_rv = hypergeom(M=M, n=P, N=round(theta * M))
            rounded_m_theta = round(theta * M)
            return(TP_rv.pmf(round_if_close((y * (P + rounded_m_theta)) / (1 + y))))

        def given_x_function(x, theta):
            rounded_m_theta = round(theta * M)
            if P + rounded_m_theta - x == 0:
                return(0)
            else:
                return(x / (P + rounded_m_theta - x))

        @add_check_theta_generator(measure)
        def expectation_function(theta):
            rounded_m_theta = round(theta * M)
            TP_rv = hypergeom(M=M, n=P, N=round(theta * M))
            return(sum([TP_rv.pmf(x) * given_x_function(x, theta) for x in range(0, min((P + 1, rounded_m_theta + 1)))]))

        @add_check_theta_generator(measure)
        def variance_function(theta):
            rounded_m_theta = round(theta * M)
            TP_rv = hypergeom(M=M, n=P, N=round(theta * M))
            return(sum([TP_rv.pmf(x) * (given_x_function(x, theta) ** 2) for x in range(0, min((P + 1, rounded_m_theta + 1)))]))

    if (measure in name_dictionary['PT']):
        @add_check_theta_generator(measure)
        def pmf_Y(y, theta):
            TP_rv = hypergeom(M=M, n=P, N=round(theta * M))
            rounded_m_theta = round(theta * M)
            return(TP_rv.pmf(round_if_close((rounded_m_theta * P * ((y - 1) ** 2)) / (M * (y ** 2) - 2 * P * y + P))))

        def given_x_function(x, theta):
            rounded_m_theta = round(theta * M)
            help_1 = x / P
            help_2 = (x - rounded_m_theta) / N
            if help_1 + help_2 == 0:
                return(0)
            else:
                return((math.sqrt(help_1 * (- help_2)) + help_2) / (help_1 + help_2))

        @add_check_theta_generator(measure)
        def expectation_function(theta):
            rounded_m_theta = round(theta * M)
            TP_rv = hypergeom(M=M, n=P, N=round(theta * M))
            return(sum([TP_rv.pmf(x) * given_x_function(x, theta) for x in range(0, min((P + 1, rounded_m_theta + 1)))]))

        @add_check_theta_generator(measure)
        def variance_function(theta):
            rounded_m_theta = round(theta * M)
            TP_rv = hypergeom(M=M, n=P, N=round(theta * M))
            return(sum([TP_rv.pmf(x) * (given_x_function(x, theta) ** 2) for x in range(0, min((P + 1, rounded_m_theta + 1)))]))

    if (measure in select_names(['G2', 'TS', 'PT'])):
        return_functions['Distribution'] = pmf_Y
        return_functions['Expectation Function'] = expectation_function
        return_functions['Variance Function'] = variance_function
        return_functions['Domain'] = generate_domain_function_given_x(
            given_x_function)

    if (measure in all_names_except(['G2', 'TS', 'PT'])):
        return_functions['Distribution'] = generate_hypergeometric_distribution(
            a, b)
        return_functions['Domain'] = generate_domain_function(a, b)
        return_functions['Fast Expectation Function'] = generate_fast_expectation_function(
            expectation_string)
        return_functions['Variance Function'] = generate_variance_function(a)
        return_functions['Expectation Function'] = generate_expectation_function(
            a, b)

    return(return_functions)


def basic_baseline_given_theta(theta, true_labels, measure, beta=1):
    """
    This function determines the mean and variance of the baseline for a given `theta` using `basic_baseline`.

    Args:
    --------
        theta (float): Parameter for the shuffle baseline.

        true_labels (list): 1-dimensional boolean list containing the true labels.

        measure (string): Measure name, see `all_names_except([''])` for possible measure names.

        beta (float): Default is 1. Parameter for the F-beta score.

    Returns:
    --------
        dict: Containing `Mean` and `Variance`

            - `Mean` (float): Expected baseline given `theta`.

            - `Variance` (float): Variance baseline given `theta`.

    See also:
    --------
        basic_baseline

    Example:
    --------
        >>> import random
        >>> random.seed(123) # To ensure similar outputs
        >>> true_labels = random.choices((0,1), k = 10000, weights = (0.9, 0.1))
        >>> baseline = basic_baseline_given_theta(theta= 0.9,true_labels= true_labels, measure= 'FBETA', beta= 1)
        >>> print('Mean: {:06.4f} and Variance: {:06.4f}'.format(baseline['Mean'], baseline['Variance']))
        Mean: 0.1805 and Variance: 0.0000
    """

    baseline = basic_baseline(true_labels=true_labels,
                              measure=measure, beta=beta)
    return({'Mean': baseline['Expectation Function'](theta), 'Variance': baseline['Variance Function'](theta)})


# %%
