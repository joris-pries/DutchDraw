
# %%
import random 
random.seed(123) # To ensure similar outputs

predicted_labels = random.choices((0,1), k = 10000, weights = (0.99, 0.1))
true_labels = random.choices((0,1), k = 10000, weights = (0.99, 0.1))
import BinaryBaselines as bbl

# Measuring markedness (MK):
print('Markedness: {:06.4f}'.format(bbl.measure_score(true_labels, predicted_labels, measure = 'MK')))

# Measuring FBETA for beta = 2:
print('F2 Score: {:06.4f}'.format(bbl.measure_score(true_labels, predicted_labels, measure = 'FBETA', beta = 2)))
# %%

results_baseline = bbl.basic_baseline_statistics(theta = 0.5, true_labels = true_labels, measure = 'FBETA', beta = 2)

print('Mean: {:06.4f}'.format(results_baseline['Mean']))
print('Variance: {:06.4f}'.format(results_baseline['Variance']))

import matplotlib.pyplot as plt
pmf_plot = [results_baseline['Distribution'](y) for y in results_baseline['Domain']]
plt.plot(results_baseline['Domain'], pmf_plot)
plt.xlabel('Measure score')
plt.ylabel('Probability mass')
plt.show()

import numpy as np
theta_values = np.arange(0, 1, 1 / sum(true_labels))
expectation_plot = [results_baseline['Fast Expectation Function'](theta) for theta in theta_values]
plt.plot(theta_values, expectation_plot)
plt.xlabel('Theta')
plt.ylabel('Expectation')
plt.show()
# %%
optimal_baseline = bbl.optimized_basic_baseline(true_labels, measure = 'FBETA', beta = 1)

print('Max Expected Value: {:06.4f}'.format(optimal_baseline['Max Expected Value']))
print('Argmax Expected Value: {:06.4f}'.format(optimal_baseline['Argmax Expected Value']))
print('Min Expected Value: {:06.4f}'.format(optimal_baseline['Min Expected Value']))
print('Argmin Expected Value: {:06.4f}'.format(optimal_baseline['Argmin Expected Value']))
# %%
