import numpy as np
import shuffle_baseline as sb

y_true = np.random.randint(2, size = 1000)

answer = sb.optimized_basic_baseline(y_true.tolist(), "G2", beta=1)
print(answer)
