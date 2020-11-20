import datetime
import numpy as np
import shuffle_baseline as sb
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# %%
results = []
for size in range(1,50):
    print(size)
    for i in range(20):
        y_true = np.random.randint(2, size = size*10)
        start = datetime.datetime.now()
        sb.optimized_basic_baseline(y_true.tolist(), "G2", beta=1)
        time = (datetime.datetime.now() - start).total_seconds()
        results.append([size * 10, time])

df = np.array(results)
df = pd.DataFrame(df, columns = ["Size","Time"])
df.to_csv("C:/Users/Etienne/Documents/Github/BinaryBaselines/BinaryBaselines/time_size_g2.csv", index= False)

# %%

results = []
for size in range(1,1000):
    print(size)
    for i in range(20):
        y_true = np.random.randint(2, size = size*10)
        start = datetime.datetime.now()
        sb.optimized_basic_baseline(y_true.tolist(), "PT", beta=1)
        time = (datetime.datetime.now() - start).total_seconds()
        results.append([size * 10, time])

df = np.array(results)
df = pd.DataFrame(df, columns = ["Size","Time"])
df.to_csv("C:/Users/Etienne/Documents/Github/BinaryBaselines/BinaryBaselines/time_size_pt.csv", index= False)

# %%
plt.figure(figsize=(20, 20))
plt.plot(df["Size"],df["Time"], ".")   
plt.show()

# %%

df = pd.read_csv("C:/Users/Etienne/Documents/Github/BinaryBaselines/BinaryBaselines/time_size_g2.csv")

#%%
df_new = df.groupby("Size")["Time"].mean().reset_index()

# %%
def func(x, a, b):
    return a * (x ) + b

popt, pcov = curve_fit(func, df_new["Size"].values, df_new["Time"].values)

plt.figure(figsize=(20, 20))
plt.plot( df_new["Size"].values, func( df_new["Size"].values, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
#plt.plot( df_new["Size"].values, func( df_new["Size"].values, *popt), 'r-',
#         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
plt.plot(df_new["Size"].values,df_new["Time"].values)
plt.legend()
plt.show()

errors = [a - b for a,b in zip(func( df_new["Size"].values, *popt),df_new["Time"])]
plt.plot(errors)



