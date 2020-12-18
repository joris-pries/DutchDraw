# %%
import random
import numpy as np
import BinaryBaselines as bbl
import math
random.seed(123) # To ensure similar outputs

N = 6
range_P = [P for P in range(1000, 1101)]
results = []
for P in range_P:
    y_true = [0] * N + [1] * P
    random.shuffle(y_true)
    outcome = bbl.optimized_basic_baseline(y_true, 'G2')
    results.append([outcome['Max Expected Value'], outcome['Argmax Expected Value'][0]])


results_array = np.array(results)

# %%
import matplotlib.pyplot as plt
plt.plot(range_P, results_array[:,1])

# %%
np.mean(results_array[:,1])
# %%
theta_range = [theta for theta in np.arange(0, 1.01, 0.01)]
k = 6
G2 = [(1- theta ** k) * math.sqrt(theta * ((1 - theta) * k + theta) / k) for theta in theta_range]
# %%
plt.plot(theta_range, G2)

# %%
theta_range[np.argmax(G2)]
# %%

# %%
range_N = [N for N in range(1,101)]
P = 1000
results = []
for N in range_N:
    y_true = [0] * N + [1] * P
    random.shuffle(y_true)
    outcome = bbl.optimized_basic_baseline(y_true, 'G2')
    results.append([outcome['Max Expected Value'], outcome['Argmax Expected Value'][0]])


results_array = np.array(results)
# %%
plt.plot(range_N, results_array[:,0])
plt.xlabel('N')
plt.ylabel('Optimal baseline')
plt.title('Optimal baseline for (P,N) = (1100, N) (P should be large)')
plt.savefig("optimal_g2.png")
plt.close()
# %%
# %%
