from shuffle_baseline import basic_baseline_given_theta, name_dictionary, optimized_basic_baseline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math

performance_measures = name_dictionary.keys()

P = 50
N = 50
M = P + N
y_true = [1] * P + [0] * N

results = []
for measure in performance_measures:
    for theta in range(M+1):
        try:
            outcome = basic_baseline_given_theta(theta/M, y_true, measure, beta=1)["Mean"]
        except:
            outcome = np.nan
        results.append([measure, theta/M, outcome])

df = pd.DataFrame(results, columns = ["Metric","Theta","Score"])

plt.figure(figsize = (6,6))
sns.lineplot(x = "Theta",y = "Score", hue = "Metric", data = df[df["Metric"].isin(["TP","TN","FN","FP"])])
plt.title(r"Expectation Base Measures over different $\theta$ with P:" + str(P) + " and N:" + str(N))
plt.xlabel(r"$\theta_{Shuffle}$")
plt.ylabel("Amount")
plt.show()

plt.figure(figsize = (6,6))
sns.lineplot(x = "Theta",y = "Score", hue = "Metric", data = df[df["Metric"].isin(['TPR', 'TNR', 'FPR', 'FNR', 'PPV', 'NPV', 'FDR', 'FOR', 'ACC'])])
plt.title(r"Expectation Base Metrics over different $\theta$ with P:" + str(P) + " and N:" + str(N))
plt.xlabel(r"$\theta_{Shuffle}$")
plt.ylabel("Score")
plt.show()

plt.figure(figsize = (6,6))
sns.lineplot(x = "Theta",y = "Score", hue = "Metric", data = df[df["Metric"].isin(['BACC', 'FBETA', 'MCC', 'BM', 'MK', 'COHEN', 'G1', 'G2', 'TS', 'PT'])])
plt.title(r"Expectation Higher Order Metrics over different $\theta$ with P:" + str(P) + " and N:" + str(N))
plt.xlabel(r"$\theta_{Shuffle}$")
plt.ylabel("Score")
plt.show()

df_pivot = pd.pivot_table(data = df.round({"Theta":2}), values = "Score", columns = "Metric", index = "Theta").round(5).abs().sort_index(ascending=False)

plt.figure(figsize = (8,8))
sns.heatmap(df_pivot[["TP","TN","FP","FN"]])
plt.title(r"Expectation Base Measures over different $\theta$ with P:" + str(P) + " and N:" + str(N))
plt.ylabel(r"$\theta_{Shuffle}$")
plt.show()

plt.figure(figsize = (8,8))
sns.heatmap(df_pivot[['TPR', 'TNR', 'FPR', 'FNR', 'PPV', 'NPV', 'FDR', 'FOR', 'ACC']])
plt.title(r"Expectation Base Metrics over different $\theta$ with P:" + str(P) + " and N:" + str(N))
plt.ylabel(r"$\theta_{Shuffle}$")
plt.show()

plt.figure(figsize = (8,8))
sns.heatmap(df_pivot[['BACC', 'FBETA', 'MCC', 'BM', 'MK', 'COHEN', 'G1', 'G2', 'TS', 'PT']])
plt.title(r"Expectation Higher Order Metrics over different $\theta$ with P:" + str(P) + " and N:" + str(N))
plt.ylabel(r"$\theta_{Shuffle}$")
plt.show()

argmax_df_pivot = df_pivot.copy()

for measure in performance_measures:
    minimum = argmax_df_pivot[measure].min()
    lb = math.floor(minimum)
    maximum = argmax_df_pivot[measure].max()
    if minimum == maximum:
        argmax_df_pivot.loc[argmax_df_pivot[measure] == minimum, measure] = lb - 3
    else:
        argmax_df_pivot.loc[argmax_df_pivot[measure] == minimum, measure] = lb - 4
        argmax_df_pivot.loc[argmax_df_pivot[measure] == maximum, measure] = lb - 1
        argmax_df_pivot.loc[((argmax_df_pivot[measure] > minimum) & (argmax_df_pivot[measure] < maximum)), measure] = lb - 2
    argmax_df_pivot[measure] = argmax_df_pivot[measure].fillna(lb - 5)
    argmax_df_pivot[measure] = argmax_df_pivot[measure] - lb
    
plt.figure(figsize = (8,6))
cmap = sns.color_palette("Spectral", 5) 
cmap[3] = (232/255,232/255,232/255)
ax = sns.heatmap(argmax_df_pivot, cmap = cmap,linewidths=.01)
colorbar = ax.collections[0].colorbar 
colorbar.set_ticks([colorbar.vmin + 4 / 5 * (0.5 + i) for i in range(5)])
colorbar.set_ticklabels(["N.D.","Min","Min = Max","None","Max"])
plt.title(r"Expectation Base Measures over different $\theta$ with P:" + str(P) + " and N:" + str(N))
plt.ylabel(r"$\theta_{Shuffle}$")
plt.xlabel("Measure/Metric")
plt.show()


M = 100
results = []
for measure in performance_measures:
    for p in range(101):
        y_true = [1] * int(p * 0.01 * M) + [0] * int((100 - p) * 0.01 * M)
        try:
            outcome = optimized_basic_baseline(y_true, measure)
        except:
            outcome = {'Max Expected Value': np.nan,
                       'Min Expected Value': np.nan}
        results.append([measure, p, M, outcome["Min Expected Value"], outcome["Max Expected Value"]])
        

df = pd.DataFrame(results, columns = ["Metric","p","M","Min Expected Value", "Max Expected Value"])
df = df[~df["Metric"].isin(["TP","TN","FN","FP"])]


plt.figure(figsize = (8,8))
sns.lineplot(x = "p",y = "Max Expected Value", hue = "Metric", data = df)
plt.title(r"Expectation Base Measures over different $\theta$ with M:" + str(M) )
plt.show()

df_pivot = pd.pivot_table(data = df, values = "Min Expected Value", columns = "Metric", index = "p").round(5).abs()

plt.figure(figsize = (6,6))
sns.heatmap(df_pivot)
plt.show()



