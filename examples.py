# %%
import BinaryBaselines as bbl
import random
import numpy as np

random.seed(123) # To ensure similar outputs

# Generate true and predicted labels
predicted_labels = random.choices((0,1), k = 10000, weights = (0.9, 0.1))
true_labels = random.choices((0,1), k = 10000, weights = (0.9, 0.1))

######################################################
# Example function: measure_score
print('\033[94mExample function: `measure_score`\033[0m')
# Measuring markedness (MK):
print('Markedness: {:06.4f}'.format(bbl.measure_score(true_labels, predicted_labels, measure = 'MK')))

# Measuring FBETA for beta = 2:
print('F2 Score: {:06.4f}'.format(bbl.measure_score(true_labels, predicted_labels, measure= 'FBETA', beta = 2)))

print('')
######################################################
# Example function: basic_baseline_given_theta
print('\033[94mExample function: `basic_baseline_given_theta`\033[0m')
results_baseline = bbl.basic_baseline_given_theta(theta = 0.5, true_labels = true_labels, measure = 'FBETA', beta = 2)

print('Mean: {:06.4f}'.format(results_baseline['Mean']))
print('Variance: {:06.4f}'.format(results_baseline['Variance']))

print('')
######################################################
# Example function: basic_baseline
print('\033[94mExample function: `basic_baseline`\033[0m')
baseline = bbl.basic_baseline(true_labels = true_labels, measure = 'G2')
print(baseline.keys())


# Expected Value
import matplotlib.pyplot as plt
theta_values = np.arange(0, 1 + 0.01, 0.01)
expected_value_plot = [baseline['Expectation Function'](theta) for theta in theta_values]
plt.plot(theta_values, expected_value_plot)
plt.xlabel('Theta')
plt.ylabel('Expected value')
#plt.savefig('expected_value_function_example.png', dpi= 600)
plt.show()

# Variance
theta_values = np.arange(0, 1 + 0.01, 0.01)
variance_plot = [baseline['Variance Function'](theta) for theta in theta_values]
plt.plot(theta_values, variance_plot)
plt.xlabel('Theta')
plt.ylabel('Variance')
#plt.savefig('variance_function_example.png', dpi= 600)
plt.show()

# Distribution and Domain
theta = 0.5
pmf_values = [baseline['Distribution'](y, theta) for y in baseline['Domain'](theta)]
plt.plot(baseline['Domain'](theta), pmf_values)
plt.xlabel('Measure score')
plt.ylabel('Probability mass')
#plt.savefig('pmf_example.png', dpi= 600)
plt.show()

print('')
######################################################
# Example function: optimized_basic_baseline
print('\033[94mExample function: `optimized_basic_baseline`\033[0m')
optimal_baseline = bbl.optimized_basic_baseline(true_labels, measure = 'FBETA', beta = 1)

print('Max Expected Value: {:06.4f}'.format(optimal_baseline['Max Expected Value']))
print('Argmax Expected Value: {:06.4f}'.format(*optimal_baseline['Argmax Expected Value']))
print('Min Expected Value: {:06.4f}'.format(optimal_baseline['Min Expected Value']))
print('Argmin Expected Value: {:06.4f}'.format(*optimal_baseline['Argmin Expected Value']))
