# %%
import math
from functools import wraps
import numpy as np
from scipy.stats import hypergeom
from tqdm import tqdm
import time
import sys


P = 20
N = 100
M = P + N

# oracle K samples
def oracle_score(K):
    y_rv = hypergeom(M=M, n=N, N=K)
    expectation = 0
    for y in range(K + 1):
        TP = P
        FN = 0
        FP = N - y
        beta_squared = 1 ** 2
        Fbeta = ((1 + beta_squared) * TP) / ((1 + beta_squared) * TP + beta_squared * FN + FP)
        expectation += y_rv.pmf(y) * Fbeta
    return expectation

K_range = range(M)
prob_list = [100 * K / M for K in K_range]
expected_oracle_baseline = [oracle_score(K) for K in K_range]

# %%
import matplotlib.pyplot as plt
plt.axhline(y=line_1, color='purple', linestyle='-')
plt.axhline(y=line_2, color='orange', linestyle='-')
plt.ylim([min(expected_oracle_baseline), max(expected_oracle_baseline)])


plt.xlabel('Percentage Oracle')
plt.ylabel('Performance score')

line_1 = 0.5
line_2 = 0.9


point_1 = prob_list[np.argmax([oracle >= line_1 for oracle in expected_oracle_baseline])]

point_2 = prob_list[np.argmax([oracle >= line_2 for oracle in expected_oracle_baseline])]

plt.savefig("demo_oracle_1.jpg")
#plt.show()



plt.plot(prob_list, expected_oracle_baseline)

plt.savefig("demo_oracle_2.jpg")
#plt.show()



plt.plot([point_1, point_2], [line_1, line_2], 'ro')

plt.axvline(x=point_1, color='red', linestyle='--')
plt.axvline(x=point_2, color='red', linestyle='--')

extraticks = [point_1, point_2]
plt.xticks(extraticks)

plt.savefig("demo_oracle_3.jpg")
#plt.show()




# %%
print(point_2 - point_1)
# %%
