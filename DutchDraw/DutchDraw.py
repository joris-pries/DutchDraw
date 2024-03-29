# %%
#import random
#import statistics
import math
from functools import wraps
import numpy as np
from scipy.stats import hypergeom
from tqdm import tqdm
import time
import sys

__all__ = ['select_all_names_except', 'baseline_functions', 'baseline_functions_given_theta',
           'measure_score', 'measure_dictionary', 'optimized_baseline_statistics',
           'round_if_close', 'select_names', 'baseline', 'classifier']

# %%

measure_dictionary = {
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
    'J': ['BM', 'BOOKMAKER INFORMEDNESS', 'INFORMEDNESS',
          'YOUDEN’S J STATISTIC', 'J'],
    'MK': ['MARKEDNESS', 'MK'],
    'KAPPA': ['COHEN', 'COHENS KAPPA', 'KAPPA'],
    'FM': ['GMEAN1', 'G MEAN 1', 'G1', 'FOWLKES-MALLOWS',
           'FOWLKES MALLOWS', 'FOWLKES', 'MALLOWS', 'FM'],
    'G2': ['GMEAN2', 'G MEAN 2', 'G2'],
    'TS': ['THREAT SCORE', 'CRITICAL SUCCES INDEX', 'TS', 'CSI']
}


def select_names(name_keys):
    """
    This function creates a list of names using the name_keys as keys for the name dictionary.
    """
    return sum([measure_dictionary[key_name] for key_name in name_keys], [])


def select_all_names_except(name_keys):
    """
    This function creates a list of all names, except the names with name_keys
    as key in the name dictionary.
    """
    return sum([list_names for key_name, list_names in measure_dictionary.items()
                if key_name not in name_keys], [])


def measure_score(y_true, y_pred, measure, beta=1):
    """
    To determine the performance of a predictive model a measure is used.
    This function determines the measure for the given input labels.

    Args:
    --------
        y_true (list or numpy.ndarray): 1-dimensional boolean list/numpy.ndarray containing the true labels.

        y_pred (list or numpy.ndarray): 1-dimensional boolean list/numpy.ndarray containing the predicted labels.

        measure (string): Measure name, see `select_all_names_except([''])` for possible measure names.

        beta (float): Default is 1. Parameter for the F-beta score.

    Returns:
    --------
        float: The score of the given measure evaluated with the predicted and true labels.

    Raises:
    --------
        ValueError
            If `measure` is not in `select_all_names_except([''])`.
        ValueError
            If `y_true` or `y_pred` does not only contain zeros and ones.

    See also:
    --------
        select_all_names_except

    Example:
    --------
        >>> import random
        >>> random.seed(123) # To ensure similar outputs
        >>> y_pred = random.choices((0, 1), k=10000, weights=(0.9, 0.1))
        >>> y_true = random.choices((0, 1), k=10000, weights=(0.9, 0.1))
        >>> print('Markedness: {:06.4f}'.format(measure_score(y_true, y_pred, measure='MK'))) # Measuring markedness (MK)
        Markedness: 0.0061
        >>> print('F2 Score: {:06.4f}'.format(measure_score(y_true, y_pred, measure='FBETA', beta=2))) # Measuring FBETA for beta = 2
        F2 Score: 0.1053

    """
    measure = measure.upper()

    # convert np.array to list
    if isinstance(y_true, np.ndarray):
        y_true = y_true.tolist()

    if isinstance(y_pred, np.ndarray):
        y_pred = y_pred.tolist()

    if measure not in select_all_names_except(['']):
        raise ValueError("This measure name is not recognized.")

    if np.unique(np.array(y_true)) not in np.array([0, 1]):
        raise ValueError("y_true should only contain zeros and ones.")

    if np.unique(np.array(y_pred)) not in np.array([0, 1]):
        raise ValueError("y_pred should only contain zeros and ones.")

    P = np.int64(sum(y_true))
    M = np.int64(len(y_true))
    N = np.int64(M - P)
    P_predicted = sum(y_pred)

    TP = np.dot(y_true, y_pred)
    FP = P_predicted - TP
    FN = P - TP
    TN = N - FP

    if measure in measure_dictionary['TP']:
        return TP

    if measure in measure_dictionary['TN']:
        return TN

    if measure in measure_dictionary['FP']:
        return FP

    if measure in measure_dictionary['FN']:
        return FN

    if measure in measure_dictionary['TPR']:
        return TP / P

    if measure in measure_dictionary['TNR']:
        return TN / N

    if measure in measure_dictionary['FPR']:
        return FP / N

    if measure in measure_dictionary['FNR']:
        return FN / P

    if measure in measure_dictionary['PPV']:
        return TP / (TP + FP)

    if measure in measure_dictionary['NPV']:
        return TN / (TN + FN)

    if measure in measure_dictionary['FDR']:
        return FP / (TP + FP)

    if measure in measure_dictionary['FOR']:
        return FN / (TN + FN)

    if measure in measure_dictionary['ACC']:
        return (TP + TN) / M

    if measure in measure_dictionary['BACC']:
        TPR = TP / P
        TNR = TN / N
        return (TPR + TNR) / 2

    if measure in measure_dictionary['FBETA']:
        beta_squared = beta ** 2
        return (1 + beta_squared) * TP / (((1 + beta_squared) * TP) + (beta_squared * FN) + FP)

    if measure in measure_dictionary['MCC']:
        return (TP * TN - FP * FN)/(math.sqrt((TP + FP) * (TN + FN) * P * N))

    if measure in measure_dictionary['J']:
        TPR = TP / P
        TNR = TN / N
        return TPR + TNR - 1

    if measure in measure_dictionary['MK']:
        PPV = TP / (TP + FP)
        NPV = TN / (TN + FN)
        return PPV + NPV - 1

    if measure in measure_dictionary['KAPPA']:
        P_o = (TP + TN) / M
        P_yes = ((TP + FP) / M) * (P / M)
        P_no = ((TN + FN) / M) * (N / M)
        P_e = P_yes + P_no
        return (P_o - P_e) / (1 - P_e)

    if measure in measure_dictionary['FM']:
        TPR = TP / P
        PPV = TP / (TP + FP)
        return math.sqrt(TPR * PPV)

    if measure in measure_dictionary['G2']:
        TPR = TP / P
        TNR = TN / N
        return math.sqrt(TPR * TNR)

    if measure in measure_dictionary['TS']:
        return TP / (TP + FN + FP)


def optimized_baseline_statistics(y_true, measure, beta=1, M_known = True, P_known = True):
    """
    This function determines the optimal `theta` that maximizes or minimizes
    the measure on the `y_true`. It also determines the corresponding extreme value.

    Args:
    --------
        y_true (list or numpy.ndarray): 1-dimensional boolean list/numpy.ndarray containing the true labels.

        measure (string): Measure name, see `select_all_names_except([''])` for possible measure names.

        beta (float): Default is 1. Parameter for the F-beta score.

        M_known (bool): True if knowledge of the number of samples can be used in determining optimality.

        P_known (bool): True if knowledge of the number of positive labels can be used in determining optimality.

    Returns:
    --------
        dict: Containing `Max Expected Value`, `Argmax Expected Value`, `Min Expected Value` and `Argmin Expected Value`.

            - `Max Expected Value` (float): Maximum of the expected values for all `theta`.

            - `Argmax Expected Value` (list): List of all `theta_star` values that maximize the expected value.

            - `Min Expected Value` (float): Minimum of the expected values for all `theta`.

            - `Argmin Expected Value` (list): List of all `theta_star` values that minimize the expected value.

    Raises:
    --------
        ValueError
            If the combination of M_known, P_known and measure leads to no known statistics.
        ValueError
            If `measure` is not in `select_all_names_except([''])`.
        ValueError
            If `y_true` does not only contain zeros and ones.

    See also:
    --------
        select_all_names_except
        baseline_functions


    Example:
    --------
        >>> import random
        >>> random.seed(123) # To ensure similar outputs
        >>> y_true = random.choices((0, 1), k=10000, weights=(0.9, 0.1))
        >>> optimal_baseline = optimized_baseline_statistics(y_true, measure='FBETA', beta=1)
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

    if return_baseline_information(measure, M_known, P_known) == False:
        raise ValueError("No known statistics in this case.")

    # convert np.array to list
    if isinstance(y_true, np.ndarray):
        y_true = y_true.tolist()

    if measure not in select_all_names_except(['']):
        raise ValueError("This measure name is not recognized.")

    if np.unique(np.array(y_true)) not in np.array([0, 1]):
        raise ValueError("y_true should only contain zeros and ones.")

    P = sum(y_true)
    M = len(y_true)
    N = M - P
    return_statistics = {}

    if measure in measure_dictionary['TP']:
        return_statistics['Max Expected Value'] = P
        return_statistics['Argmax Expected Value'] = [1]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [0]

    if measure in measure_dictionary['TN']:
        return_statistics['Max Expected Value'] = N
        return_statistics['Argmax Expected Value'] = [0]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [1]

    if measure in measure_dictionary['FN']:
        return_statistics['Max Expected Value'] = P
        return_statistics['Argmax Expected Value'] = [0]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [1]

    if measure in measure_dictionary['FP']:
        return_statistics['Max Expected Value'] = N
        return_statistics['Argmax Expected Value'] = [1]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [0]

    if measure in measure_dictionary['TPR']:
        return_statistics['Max Expected Value'] = 1
        return_statistics['Argmax Expected Value'] = [1]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [0]

    if measure in measure_dictionary['TNR']:
        return_statistics['Max Expected Value'] = 1
        return_statistics['Argmax Expected Value'] = [0]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [1]

    if measure in measure_dictionary['FNR']:
        return_statistics['Max Expected Value'] = 1
        return_statistics['Argmax Expected Value'] = [0]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [1]

    if measure in measure_dictionary['FPR']:
        return_statistics['Max Expected Value'] = 1
        return_statistics['Argmax Expected Value'] = [1]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [0]

    if measure in measure_dictionary['PPV']:
        return_statistics['Max Expected Value'] = P/M
        return_statistics['Argmax Expected Value'] = [
            i/M for i in range(1, M + 1)]
        return_statistics['Min Expected Value'] = P/M
        return_statistics['Argmin Expected Value'] = [
            i/M for i in range(1, M + 1)]

    if measure in measure_dictionary['NPV']:
        return_statistics['Max Expected Value'] = N/M
        return_statistics['Argmax Expected Value'] = [i/M for i in range(0, M)]
        return_statistics['Min Expected Value'] = N/M
        return_statistics['Argmin Expected Value'] = [i/M for i in range(0, M)]

    if measure in measure_dictionary['FDR']:
        return_statistics['Max Expected Value'] = N/M
        return_statistics['Argmax Expected Value'] = [
            i/M for i in range(1, M + 1)]
        return_statistics['Min Expected Value'] = N/M
        return_statistics['Argmin Expected Value'] = [
            i/M for i in range(1, M + 1)]

    if measure in measure_dictionary['FOR']:
        return_statistics['Max Expected Value'] = P/M
        return_statistics['Argmax Expected Value'] = [i/M for i in range(0, M)]
        return_statistics['Min Expected Value'] = P/M
        return_statistics['Argmin Expected Value'] = [i/M for i in range(0, M)]

    if measure in measure_dictionary['FBETA']:
        beta_squared = beta ** 2
        return_statistics['Max Expected Value'] = (
            1 + beta_squared) * P / (beta_squared * P + M)
        return_statistics['Argmax Expected Value'] = [1]
        return_statistics['Min Expected Value'] = (1 + beta_squared) * P / (M * (beta_squared * P + 1))
        return_statistics['Argmin Expected Value'] = [1/M]

    if measure in measure_dictionary['J']:
        return_statistics['Max Expected Value'] = 0
        return_statistics['Argmax Expected Value'] = [
            i/M for i in range(0, M + 1)]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [
            i/M for i in range(0, M + 1)]

    if measure in measure_dictionary['MK']:
        return_statistics['Max Expected Value'] = 0
        return_statistics['Argmax Expected Value'] = [i/M for i in range(1, M)]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [i/M for i in range(1, M)]

    if measure in measure_dictionary['ACC']:
        return_statistics['Max Expected Value'] = max((N/M, P/M))
        return_statistics['Min Expected Value'] = min((N/M, P/M))
        if P == N:
            return_statistics['Argmax Expected Value'] = [i/M for i in range(0, M+1)]
            return_statistics['Argmin Expected Value'] = [i/M for i in range(0, M+1)]
        else:
            return_statistics['Argmax Expected Value'] = [int((P >= N))]
            return_statistics['Argmin Expected Value'] = [int((P < N))]

    if measure in measure_dictionary['BACC']:
        return_statistics['Max Expected Value'] = 0.5
        return_statistics['Argmax Expected Value'] = [i/M for i in range(0, M+1)]
        return_statistics['Min Expected Value'] = 0.5
        return_statistics['Argmin Expected Value'] = [i/M for i in range(0, M+1)]

    if measure in measure_dictionary['MCC']:
        return_statistics['Max Expected Value'] = 0
        return_statistics['Argmax Expected Value'] = [i/M for i in range(1, M)]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [i/M for i in range(1, M)]

    if measure in measure_dictionary['KAPPA']:
        return_statistics['Max Expected Value'] = 0
        return_statistics['Argmax Expected Value'] = [
            i/M for i in range(0, M + 1)]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [
            i/M for i in range(0, M + 1)]

    if measure in measure_dictionary['FM']:
        return_statistics['Max Expected Value'] = math.sqrt(P / M)
        return_statistics['Argmax Expected Value'] = [1]
        return_statistics['Min Expected Value'] = math.sqrt(P) / M
        return_statistics['Argmin Expected Value'] = [1/M]

    if measure in measure_dictionary['G2']:
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [0, 1]

        result = [np.nan] * (M + 1)
        time_to_exc = round(0.000175452 * M ** 1.8841 -0.0512485)
        print("Press Control + C to stop the code")

        if time_to_exc < 60:
            print("Estimated time to execute is: " + str(time_to_exc) + " seconds." )
        else:
            time_to_exc = round(time_to_exc / 60)
            if time_to_exc < 60:
                print("Estimated time to execute is: " + str(time_to_exc) + " minutes." )
                time_to_exc = round(time_to_exc / 60)
            else:
                time_to_exc_hour = round(time_to_exc / 60)
                print("Estimated time to execute is: " + str(time_to_exc_hour) + " hours." )
        time.sleep(2)
        try:
            for i in tqdm(range(0, M + 1)):
                theta = i / M
                rounded_m_theta = round(round(M * theta))
                TP_rv = hypergeom(M=M, n=P, N=rounded_m_theta)
                result[i] = sum([(math.sqrt(k * (N - rounded_m_theta + k) / (P * N))) * TP_rv.pmf(k)
                                 if TP_rv.pmf(k) > 0 else 0 for k in range(int(max(0, rounded_m_theta - N)),
                                                                           int(min((P + 1, rounded_m_theta + 1))))])
        except KeyboardInterrupt:
            print("\nThe code is stopped.")
            print("This means that the max expected value could not be calculated.")
            print("You only get the min and argmin.")

            return_statistics['Max Expected Value'] = np.nan
            return_statistics['Argmax Expected Value'] = [np.nan]
            return return_statistics

        return_statistics['Max Expected Value'] = np.nanmax(result)
        return_statistics['Argmax Expected Value'] = [
            i/M for i, j in enumerate(result) if j == return_statistics['Max Expected Value']]

    if measure in measure_dictionary['TS']:
        return_statistics['Max Expected Value'] = P / M
        if P == 1:
            return_statistics['Argmax Expected Value'] = [i/M for i in range(1, M+1)]
        else:
            return_statistics['Argmax Expected Value'] = [1]
        return_statistics['Min Expected Value'] = 0
        return_statistics['Argmin Expected Value'] = [0]

    return return_statistics


def round_if_close(x):
    """
    This function is used to round x if it is close. This is useful for the pmf of the hypergeometric distribution.
    """
    if math.isclose(x, round(x), abs_tol=0.000001):
        return round(x)
    return x


def add_check_theta_generator(measure):
    """
    This is a decorator to add a ValueError to a function if theta is not in the proper interval.
    """
    include_0 = True
    include_1 = True
    measure = measure.upper()
    # Should 0 be included
    if measure in select_names(['PPV', 'FDR', 'MCC', 'MK', 'FM']):
        include_0 = False
    # Should 1 be included
    if measure in select_names(['NPV', 'FOR', 'MCC', 'MK']):
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


expectation_docstring = """
        Expectation function of measure.

        Args:
        --------
            theta (float): Parameter for the shuffle baseline.

        Returns:
        --------
            float: The expectation of the measure given `theta`.
        """

pmf_docstring = """
            Probability mass function of measure.

            Args:
            --------
                y (float): measure score

                theta (float): Parameter for the shuffle baseline.

            Returns:
            --------
                float: The probability that the measure is `y` using the shuffle approach.
            """

variance_docstring = """
            Variance function of measure.

            Args:
            --------
                theta (float): Parameter for the shuffle baseline.

            Returns:
            --------
                float: The variance of the measure given `theta`.
            """

fast_expectation_docstring = """
            Fast expectation function of measure.

            Args:
            --------
                theta (float): Parameter for the shuffle baseline.

            Returns:
            --------
                float: The fast expectation of the measure given `theta`.
            """

domain_docstring = """
            Domain function of measure. All scores with non-zero probability.

            Args:
            --------
                theta (float): Parameter for the shuffle baseline.

            Returns:
            --------
                list: List of all scores with non-zero probability.
            """


def add_docstring(docstring):
    """
    This function is used to set a docstring of a function
    """
    def _add_docstring(func):
        func.__doc__ = docstring
        return func
    return _add_docstring


def baseline_functions(y_true, measure, beta=1, M_known = True, P_known = True):
    """
    This function returns a dictionary of functions that can be used to determine
    statistics (such as expectation and variance) for all possible values of `theta`.

    Args:
    --------
        y_true (list or numpy.ndarray): 1-dimensional boolean list/numpy.ndarray containing the true labels.

        measure (string): Measure name, see `select_all_names_except([''])` for possible measure names.

        beta (float): Default is 1. Parameter for the F-beta score.

        M_known (bool): True if knowledge of the number of samples can be used in determining optimality.

        P_known (bool): True if knowledge of the number of positive labels can be used in determining optimality.

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
            If the combination of M_known, P_known and measure leads to no known statistics.
        ValueError
            If `measure` is not in `select_all_names_except([''])`.
        ValueError
            If `y_true` does not only contain zeros and ones.

    See also:
    --------
        select_all_names_except
        select_names
        round_if_close

    Example:
    --------
        >>> import random
        >>> random.seed(123) # To ensure similar outputs
        >>> y_true = random.choices((0, 1), k=10000, weights=(0.9, 0.1))
        >>> baseline = baseline_functions(y_true, 'MK')
        >>> print(baseline.keys())
        dict_keys(['Distribution', 'Domain', 'Fast Expectation Function', 'Variance Function', 'Expectation Function'])
    """

    measure = measure.upper()

    # convert np.array to list
    if isinstance(y_true, np.ndarray):
        y_true = y_true.tolist()

    if measure not in select_all_names_except(['']):
        raise ValueError("This measure name is not recognized.")

    if np.unique(np.array(y_true)) not in np.array([0, 1]):
        raise ValueError("y_true should only contain zeros and ones.")

    P = sum(y_true)
    M = len(y_true)
    N = M - P

    # Used to return all functions
    return_functions = {}

    # Used to generate pmf functions
    def generate_hypergeometric_distribution(a, b):
        @add_docstring(pmf_docstring)
        @add_check_theta_generator(measure)
        def pmf_Y(y, theta):
            TP_rv = hypergeom(M=M, n=P, N=round(theta * M))
            # Use round_if_close function, because of small computation errors in python
            return TP_rv.pmf(round_if_close((y - b) / a))
        return pmf_Y

    # Used to generate variance functions
    def generate_variance_function(a):
        @add_docstring(variance_docstring)
        @add_check_theta_generator(measure)
        def variance_function(theta):
            theta_star = round(theta * M) / M
            rounded_m_theta = round(theta * M)
            var_tp = (theta_star * (1 - theta_star) * P * N) / (M - 1)
            return (eval(a) ** 2) * var_tp
        return variance_function

    # Used to generate expectation functions
    def generate_expectation_function(a, b):
        @add_docstring(expectation_docstring)
        @add_check_theta_generator(measure)
        def expectation_function(theta):
            theta_star = round(theta * M) / M
            rounded_m_theta = round(theta * M)
            mean_tp = theta_star * P
            return eval(a) * mean_tp + eval(b)
        return expectation_function

    # Used to generate fast expectation functions. The expectation string is used to alter the function.
    def generate_fast_expectation_function(expectation_string):
        @add_docstring(fast_expectation_docstring)
        @add_check_theta_generator(measure)
        def fast_expectation_function(theta):
            theta_star = round(theta * M) / M
            return eval(expectation_string)
        return fast_expectation_function

    # Used to generate domain functions
    def generate_domain_function(a, b):
        @add_docstring(domain_docstring)
        @add_check_theta_generator(measure)
        def domain_function(theta):
            theta_star = round(theta * M) / M
            rounded_m_theta = round(theta * M)
            return [(eval(a) * x) + eval(b) for x in range(int(max(0, rounded_m_theta - N)), int(min((P + 1, rounded_m_theta + 1))))]
        return domain_function

    # Used to generate domain function for TS and G2.
    def generate_domain_function_given_x(given_x_function):
        @add_check_theta_generator(measure)
        def domain_function(theta):
            rounded_m_theta = round(theta * M)
            return np.unique([given_x_function(x, theta) for x in range(int(max(0, rounded_m_theta - N)), int(min((P + 1, rounded_m_theta + 1))))])
        return domain_function

    if measure in measure_dictionary['TP']:
        a = '1'
        b = '0'
        expectation_string = 'theta_star * ' + str(P)

    if measure in measure_dictionary['TN']:
        a = '1'
        b = str(N) + ' - rounded_m_theta'
        expectation_string = '(1 - theta_star) * ' + str(N)

    if measure in measure_dictionary['FP']:
        a = '-1'
        b = 'rounded_m_theta'
        expectation_string = 'theta_star * ' + str(N)

    if measure in measure_dictionary['FN']:
        a = '-1'
        b = str(P)
        expectation_string = '(1 - theta_star) * ' + str(P)

    if measure in measure_dictionary['TPR']:
        a = '1 / ' + str(P)
        b = '0'
        expectation_string = 'theta_star'

    if measure in measure_dictionary['TNR']:
        a = '1 / ' + str(N)
        b = '(' + str(N) + ' - rounded_m_theta) / ' + str(N)
        expectation_string = '1 - theta_star'

    if measure in measure_dictionary['FPR']:
        a = '-1 / ' + str(N)
        b = 'rounded_m_theta / ' + str(N)
        expectation_string = 'theta_star'

    if measure in measure_dictionary['FNR']:
        a = '-1 / ' + str(P)
        b = '1'
        expectation_string = '1 - theta_star'

    if measure in measure_dictionary['PPV']:
        a = '1 / rounded_m_theta'
        b = '0'
        expectation_string = str(P) + ' / ' + str(M)

    if measure in measure_dictionary['NPV']:
        a = '1 / (' + str(M) + ' - rounded_m_theta)'
        b = '(' + str(N) + ' - rounded_m_theta) / (' + \
            str(M) + ' - rounded_m_theta)'
        expectation_string = str(N) + ' / ' + str(M)

    if measure in measure_dictionary['FDR']:
        a = '-1 / rounded_m_theta'
        b = '1'
        expectation_string = str(N) + ' / ' + str(M)

    if measure in measure_dictionary['FOR']:
        a = '-1 / (' + str(M) + ' - rounded_m_theta)'
        b = '1 - ((' + str(N) + ' - rounded_m_theta) / (' + \
            str(M) + ' - rounded_m_theta))'
        expectation_string = str(P) + ' / ' + str(M)

    if measure in measure_dictionary['ACC']:
        a = '2 / ' + str(M)
        b = '(' + str(N) + ' - rounded_m_theta) / ' + str(M)
        expectation_string = '((1 - theta_star) * ' + str(N) + \
            ' + (theta_star * ' + str(P) + ')) / ' + str(M)

    if measure in measure_dictionary['BACC']:
        a = '(1 / (2 * ' + str(P) + ')) + (1 / (2 * ' + str(N) + '))'
        b = '(' + str(N) + ' - rounded_m_theta) / (2 * ' + str(N) + ')'
        expectation_string = '1 / 2'

    if measure in measure_dictionary['FBETA']:
        a = '(1 + (' + str(beta) + ' ** 2)) / ((' + str(beta) + \
            ' ** 2) * ' + str(P) + ' + ' + str(M) + ' * theta_star)'
        b = '0'
        expectation_string = '((1 + (' + str(beta) + ' ** 2)) * theta_star * ' + str(
            P) + ') / (' + str(beta) + ' ** 2) * ' + str(P) + ' + ' + str(M) + ' * theta_star)'

    if measure in measure_dictionary['MCC']:
        a = '1 / (math.sqrt(theta_star * (1 - theta_star) * ' + \
            str(P) + ' * ' + str(N) + '))'
        b = '- theta_star * ' + \
            str(P) + ' / (math.sqrt(theta_star * (1 - theta_star) * ' + \
            str(P) + ' * ' + str(N) + '))'
        expectation_string = '0'

    if measure in measure_dictionary['J']:
        a = '(1 / ' + str(P) + ') + (1 / ' + str(N) + ')'
        b = '- rounded_m_theta / ' + str(N)
        expectation_string = '0'

    if measure in measure_dictionary['MK']:
        a = '(1 / rounded_m_theta) + (1 / (' + str(M) + ' - rounded_m_theta))'
        b = '-' + str(P) + ' / (' + str(M) + ' - rounded_m_theta)'
        expectation_string = '0'

    if measure in measure_dictionary['KAPPA']:
        a = '2 / ((1 - theta_star) * ' + str(P) + \
            ' + theta_star * ' + str(N) + ')'
        b = '- 2 * theta_star * ' + \
            str(P) + ' / ((1 - theta_star) * ' + \
            str(P) + ' + theta_star * ' + str(N) + ')'
        expectation_string = '0'

    if measure in measure_dictionary['FM']:
        a = '1 / (math.sqrt(' + str(P) + ' * rounded_m_theta))'
        b = '0'
        expectation_string = 'math.sqrt(theta_star * ' + \
            str(P) + ' / ' + str(M) + ')'

    if measure in measure_dictionary['G2']:
        @add_docstring(pmf_docstring)
        @add_check_theta_generator(measure)
        def pmf_Y(y, theta):
            TP_rv = hypergeom(M=M, n=P, N=round(theta * M))
            rounded_m_theta = round(theta * M)
            help_constant = math.sqrt(
                (rounded_m_theta ** 2) - 2 * rounded_m_theta * N + (N ** 2) + 4 * P * N * (y ** 2))
            value_1 = (1/2) * ((- help_constant) + rounded_m_theta - N)
            value_2 = (1/2) * (help_constant + rounded_m_theta - N)
            return TP_rv.pmf(round_if_close(value_1)) + TP_rv.pmf(round_if_close(value_2))

        def given_x_function(x, theta):
            rounded_m_theta = round(theta * M)
            return math.sqrt((x / P) * ((N - rounded_m_theta + x) / N))

        @add_docstring(expectation_docstring)
        @add_check_theta_generator(measure)
        def expectation_function(theta):
            rounded_m_theta = round(theta * M)
            TP_rv = hypergeom(M=M, n=P, N=round(theta * M))
            return sum([TP_rv.pmf(x) * given_x_function(x, theta) for x in range(int(max(0, rounded_m_theta - N)), int(min((P + 1, rounded_m_theta + 1))))])

        @add_docstring(variance_docstring)
        @add_check_theta_generator(measure)
        def variance_function(theta):
            rounded_m_theta = round(theta * M)
            TP_rv = hypergeom(M=M, n=P, N=round(theta * M))
            return sum([TP_rv.pmf(x) * (given_x_function(x, theta) ** 2) for x in range(int(max(0, rounded_m_theta - N)), int(min((P + 1, rounded_m_theta + 1))))])

    if measure in measure_dictionary['TS']:
        @add_docstring(pmf_docstring)
        @add_check_theta_generator(measure)
        def pmf_Y(y, theta):
            TP_rv = hypergeom(M=M, n=P, N=round(theta * M))
            rounded_m_theta = round(theta * M)
            return TP_rv.pmf(round_if_close((y * (P + rounded_m_theta)) / (1 + y)))

        def given_x_function(x, theta):
            rounded_m_theta = round(theta * M)
            if P + rounded_m_theta - x == 0:
                return 0
            return x / (P + rounded_m_theta - x)

        @add_docstring(expectation_docstring)
        @add_check_theta_generator(measure)
        def expectation_function(theta):
            rounded_m_theta = round(theta * M)
            TP_rv = hypergeom(M=M, n=P, N=round(theta * M))
            return sum([TP_rv.pmf(x) * given_x_function(x, theta) for x in range(int(max(0, rounded_m_theta - N)), int(min((P + 1, rounded_m_theta + 1))))])

        @add_docstring(variance_docstring)
        @add_check_theta_generator(measure)
        def variance_function(theta):
            rounded_m_theta = round(theta * M)
            TP_rv = hypergeom(M=M, n=P, N=round(theta * M))
            return sum([TP_rv.pmf(x) * (given_x_function(x, theta) ** 2) for x in range(int(max(0, rounded_m_theta - N)), int(min((P + 1, rounded_m_theta + 1))))])

    if measure in select_names(['G2', 'TS']):
        return_functions['Distribution'] = pmf_Y
        return_functions['Expectation Function'] = expectation_function
        return_functions['Variance Function'] = variance_function
        return_functions['Domain'] = generate_domain_function_given_x(
            given_x_function)

    if measure in select_all_names_except(['G2', 'TS']):
        return_functions['Distribution'] = generate_hypergeometric_distribution(
            a, b)
        return_functions['Domain'] = generate_domain_function(a, b)
        return_functions['Fast Expectation Function'] = generate_fast_expectation_function(
            expectation_string)
        return_functions['Variance Function'] = generate_variance_function(a)
        return_functions['Expectation Function'] = generate_expectation_function(
            a, b)

    return return_functions


def baseline_functions_given_theta(theta, y_true, measure, beta=1, M_known = True, P_known = True):
    """
    This function determines the mean and variance of the baseline for a given `theta` using `baseline_functions`.

    Args:
    --------
        theta (float): Parameter for the shuffle baseline.

        y_true (list or numpy.ndarray): 1-dimensional boolean list/numpy.ndarray containing the true labels.

        measure (string): Measure name, see `select_all_names_except([''])` for possible measure names.

        beta (float): Default is 1. Parameter for the F-beta score.

        M_known (bool): True if knowledge of the number of samples can be used in determining optimality.

        P_known (bool): True if knowledge of the number of positive labels can be used in determining optimality.

    Returns:
    --------
        dict: Containing `Mean` and `Variance`

            - `Mean` (float): Expected baseline given `theta`.

            - `Variance` (float): Variance baseline given `theta`.

    Raises:
    --------
        ValueError
            If the combination of M_known, P_known and measure leads to no known statistics.

    See also:
    --------
        baseline_functions

    Example:
    --------
        >>> import random
        >>> random.seed(123) # To ensure similar outputs
        >>> y_true = random.choices((0, 1), k=10000, weights=(0.9, 0.1))
        >>> baseline = baseline_functions_given_theta(theta= 0.9, y_true=y_true, measure='FBETA', beta=1)
        >>> print('Mean: {:06.4f} and Variance: {:06.4f}'.format(baseline['Mean'], baseline['Variance']))
        Mean: 0.1805 and Variance: 0.0000
    """

    baseline = baseline_functions(y_true=y_true,
                              measure=measure, beta=beta)
    return {'Mean': baseline['Expectation Function'](theta), 'Variance': baseline['Variance Function'](theta)}

# %%

def return_baseline_information(measure = '', M_known = True, P_known = True):
    if measure in select_names(['ACC']) and (P_known == False or M_known == False):
        return False
    if measure in select_names(['FM', 'FBETA']) and M_known == False and P_known == False:
        return False
    else:
        return True


def baseline(y_true, measure= '', theta = 'optimal', M_known = True, P_known = True, beta = 1):
    """
    Statistics/information about the Dutch Draw baseline, combining the functions: optimized_baseline_statistics, baseline_functions, baseline_functions_given_theta.

    Args:
    --------
        y_true (list or numpy.ndarray): 1-dimensional boolean list/numpy.ndarray containing the true labels.

        measure (string): Measure name, see `select_all_names_except([''])` for possible measure names.

        theta (float or string):

            - 'optimal' (default): statistics of the optimal baseline are returned. (See `optimized_baseline_statistics`).

            - 'all': functions of the baseline are returned for all theta. (See `baseline_functions`).

            - float: statistics of the baseline for this given `theta`. (See `baseline_functions_given_theta`).


        M_known (bool): True if knowledge of the number of samples can be used in determining optimality.

        P_known (bool): True if knowledge of the number of positive labels can be used in determining optimality.

        beta (float): Default is 1. Parameter for the F-beta score.

    Returns:
    --------
        Dependent on theta. See `optimized_baseline_statistics`, `baseline_functions` and `baseline_functions_given_theta`.

    Raises:
    --------
        ValueError
            If `M_known` is False and `P_known` is True

    See also:
    --------
        optimized_baseline_statistics
        baseline_functions
        baseline_functions_given_theta
    
    Example:
    --------
        >>> import random
        >>> random.seed(123) # To ensure similar outputs
        >>> y_true = random.choices((0, 1), k=1000, weights=(0.9, 0.1))
        >>> stats =  baseline(y_true, measure = 'ACC', theta = 'optimal')
        >>> print(stats)
        {'Max Expected Value': 0.888, 'Min Expected Value': 0.112, 'Argmax Expected Value': [0], 'Argmin Expected Value': [1]}
        >>> stats =  baseline(y_true, measure = 'FBETA', theta = 0.2)
        >>> print(stats)
        {'Mean': 0.1435897435897436, 'Variance': 0.0006545401417196289}
        >>> stats =  baseline(y_true, measure = 'TS', theta = 'all')
        >>> print(stats["Expectation Function"](0.5)) #Function depends on theta, here 0.5.
        0.10080806593812942
    """


    if M_known == False and P_known == True:
        raise ValueError("This case has not been investigated. If M is unknown, P must also be unknown.")

    if theta == 'optimal':
        return optimized_baseline_statistics(y_true, measure, beta, M_known = True, P_known = True)

    elif theta == 'all':
        return baseline_functions(y_true, measure, beta, M_known = True, P_known = True)

    else:
        return baseline_functions_given_theta(theta, y_true, measure, beta, M_known = True, P_known = True)

def generate_y_true(M, P):
    return [1] * P + [0] * (M - P)


def classifier(y_true=None, theta='max',  measure='', beta = 1,
                        M_known = True, P_known = True, E_P_x_E_N = None):
    """
    This function gives the outcome of the Dutch Draw classifier given some parameters

    Args:
    --------
        y_true (list or numpy.ndarray): 1-dimensional boolean list/numpy.ndarray containing the true labels.

        theta (float): Parameter for the shuffle baseline. Can be a float between 0 or 1 or
        it can be the optimal theta (min or max).

        measure (string): Measure name, see `select_all_names_except([''])` for possible measure names.

        beta (float): Default is 1. Parameter for the F-beta score.

        M_known (bool): True if knowledge of the number of samples can be used in determining optimality.

        P_known (bool): True if knowledge of the number of positive labels can be used in determining optimality.

        E_P_x_E_N (string): With this parameter, if we do not know P, we can still say something about P.
        The x shows whether or not the expected P is bigger (>), smaller (<) or equal (=) to the expected number of
        negatives. If this is unknown, we can set it None.

    Returns:
    --------
        y_pred: prediction 1-dimensional boolean containing predicted labels of the Dutch Draw.

    Raises:
    --------
        ValueError
            If `y_true' is not a list consisting of zeros and ones.
        ValueError
            If 'theta' is not a float between zero and one or "max" or "min".
        ValueError
            If `measure' is not considered.
        ValueError
            If `M_known' is False and `P_known' is True.
        ValueError
            If `beta' is negative.
        ValueError
            If `E_P_x_E_N' is not None, <, = or >.

    See also:
    --------
        optimized_baseline_statistics

    Example:
    --------
        >>> import random
        >>> random.seed(123) # To ensure similar outputs
        >>> y_true = random.choices((0, 1), k=1000, weights=(0.9, 0.1))
        >>> y_pred = classifier(y_true=y_true, theta = "max", measure='ACC', 
                                          P_known = False, E_P_x_E_N = ">")
        >>> print("Length y_pred:", len(y_pred), ", number of positives:", np.sum(y_pred))
        Length y_pred: 1000 , number of positives: 1000
        >>> y_pred = classifier(y_true=y_true, theta = "min", measure='TS')
        >>> print("Length y_pred:", len(y_pred), ", number of positives:", np.sum(y_pred))
        Length y_pred: 1000 , number of positives: 0
    """
    if y_true is None :
        raise ValueError("y_true must be given")

    if isinstance(y_true, np.ndarray):
        y_true = y_true.tolist()

    if np.unique(np.array(y_true)) not in np.array([0, 1]):
        raise ValueError("y_true should only contain zeros and ones.")

    if isinstance(theta, float):
        if theta < 0 or theta > 1:
            raise ValueError("theta must be between 0 and 1.")
    else:
        if not theta in ["min","max"]:
            raise ValueError("theta must be float, 'min' or 'max'.")

    if measure not in select_all_names_except(['']):
        raise ValueError("This measure name is not recognized.")

    if M_known == False and P_known == True:
        raise ValueError("This case has not been investigated. If M is unknown, P must also be unknown.")
        
    if beta < 0:
        raise ValueError("beta must be positive or 0.")

    if not E_P_x_E_N in [None, "<","=",">"]:
        raise ValueError("Variable E_P_x_E_N contains non-ommited value.")

    M = len(y_true)

    if isinstance(theta, float):
        return [1] * round(M * theta) + [0] * round(M * ( 1- theta) )

    if measure == "FM" or measure == "FBETA":
        if not M_known and not P_known:
            if theta == "max":
                return [1] * M
            if theta == "min":
                return [1] + [0] * (M - 1)

    if measure == "ACC":
        if not M_known and not P_known:
            if theta == "max":

                y_pred = []
                while len(y_pred) < M:
                    y_pred.append(0)
                    y_pred.append(1)
                return y_pred[:M]

            if theta == "min":
                return [1] * M
        if M_known and not P_known:
            if theta == "max":
                if E_P_x_E_N == None :
                    y_pred = [1] * math.ceil(M * 0.5) + [0] * math.ceil(M * 0.5)
                    return y_pred[:M]
                if E_P_x_E_N in ["<","="]:
                    return [0] * M
                if E_P_x_E_N == ">":
                    return [1] * M
            if theta == "min":
                if E_P_x_E_N in [None,">"]:
                    return [0] * M
                if E_P_x_E_N in ["<","="]:
                    return [1] * M

    if theta == "max":
        t = optimized_baseline_statistics(y_true, measure, beta)["Argmax Expected Value"][0]
    if theta == "min":
        t = optimized_baseline_statistics(y_true, measure, beta)["Argmin Expected Value"][0]
    return [1] * round(M * t) + [0] * round(M * (1 - t))


#%%

