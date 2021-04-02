# %%
import numpy as np
import random
from tqdm import tqdm
import BinaryBaselines as bbl
# %%
max_range = 1000
with open("results.txt", "w") as text_file:
    for P in tqdm(range(2, max_range)):
        text_file.write('Checking P = {}\n'.format(P))
        for N in range(1, max_range):
            M = P + N
            y_true = N * [0] + P * [1]
            theta_range = np.arange(0, 1+ 1/(2*M), 1/M)
            result_baseline = bbl.basic_baseline(y_true, 'TS')
            expectations = [result_baseline['Expectation Function'](theta) for theta in theta_range]
            arg_max_expectations = np.argmax(expectations)

            if arg_max_expectations != (len(expectations) - 1):
                # import matplotlib.pyplot as plt
                # plt.plot(theta_range, expectations)
                # plt.xlabel('Theta')
                # plt.ylabel('Expected value')
                # plt.show()
                text_file.write('Not optimal for P = {} and N = {}\n'.format(P, N))
                #break




# %%
