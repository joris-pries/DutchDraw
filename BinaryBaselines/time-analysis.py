
import datetime
import numpy as np
import shuffle_baseline as sb
import matplotlib.pyplot as plt
import pandas as pd

results = []
for size in range(1,30):
    print(size)
    for i in range(20):
        y_true = np.random.randint(2, size = size*10)
        start = datetime.datetime.now()
        sb.optimized_basic_baseline(y_true.tolist(), "G2", beta=1)
        time = (datetime.datetime.now() - start).total_seconds()
        results.append([size * 10, time])

df = np.array(results)
df = pd.DataFrame(df, columns = ["Size","Time"])
df["Size"] = df["Size"] * 10
df.to_csv("C:/Users/Etienne/Documents/Github/BinaryBaselines/BinaryBaselines/time_size_g2.csv", index= False)

df_new = df.groupby("Size")["Time"].mean().reset_index()

def func(x, a, b, c):
    return a * np.exp(-b * x) + c





plt.figure(figsize=(20, 20))
plt.plot(df["Size"],df["Time"], ".")   
plt.show()



from scipy.stats import hypergeom
import math

M = 100
P = 10
N = 90
result = [np.nan] * (M + 1)
for i in range(0, M + 1): # (M + 1)
    theta = i / M
    rounded_m_theta = round(round(M * theta))
    TP_rv = hypergeom(M=M, n=P, N=rounded_m_theta)
    lb = int(max(0, rounded_m_theta - N))
    ub = int(min((P + 1, rounded_m_theta + 1)))
    
    result[i] = sum([(math.sqrt(k * (N - rounded_m_theta + k) / (P * N))) * TP_rv.pmf(k) for k in range(lb,ub)])

M * P 