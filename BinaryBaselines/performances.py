from shuffle_baseline import basic_baseline_given_theta, name_dictionary, optimized_basic_baseline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math

performance_measures = name_dictionary.keys()

#%%

BASE_MEASURES = ["TP", "TN", "FN", "FP"]
BASE_METRICS = ['TPR', 'TNR', 'FPR', 'FNR', 'PPV', 'NPV', 'FDR', 'FOR', 'ACC']
HIGHER_ORDER_METRICS = ['BACC', 'FBETA', 'MCC', 'J', 'MK',
                        'KAPPA', 'FM', 'G2', 'TS', 'PT']

#%%
P = 100
N = 5
M = P + N
y_true = [1] * P + [0] * N

#%%
def get_baseline():
    results = []
    for measure in performance_measures:
        for theta in range(M+1):
            try:
                outcome = basic_baseline_given_theta(theta/M, y_true, measure, beta=1)["Mean"]
            except:
                outcome = np.nan
            results.append([measure, theta/M, outcome])
    df = pd.DataFrame(results, columns = ["Metric","Theta","Score"])
    return df

def plot_metrics_over_theta(df, metric_list, y_label):

    plt.figure(figsize = (6,6))
    sns.lineplot(x = "Theta",y = "Score", hue = "Metric", data =  df[df["Metric"].isin(metric_list)])
    plt.title(r"Expectation Base Measures over different $\theta$ with P:" + str(P) + " and N:" + str(N))
    plt.xlabel(r"$\theta^*$")
    plt.ylabel(y_label)
    plt.show()

def plot_heatmap(df, metric_list, P, N):
    plt.figure(figsize = (8,8))
    sns.heatmap(df[metric_list])
    plt.title(r"Expectation Base Measures over different $\theta$ with P:" + str(P) + " and N:" + str(N))
    plt.ylabel(r"$\theta^*$")
    plt.show()
    
def translate_scores_to_extrema(df):
    df_ = df.copy()

    for measure in performance_measures:
        lower = df_[measure].min()
        lower_floor = math.floor(lower)
        upper = df_[measure].max()
    
        if lower == upper:
            df_.loc[df_[measure] == lower, measure] = lower_floor - 3
        else:
            df_.loc[df_[measure] == lower, measure] = lower_floor - 4
            df_.loc[df_[measure] == upper, measure] = lower_floor - 1
            df_.loc[((df_[measure] > lower) & (df_[measure] < upper)), measure] = lower_floor - 2
        df_[measure] = df_[measure].fillna(lower_floor - 5)
        df_[measure] = df_[measure] - lower_floor
    return df_

def plot_heatmap_extrema(df):
    plt.figure(figsize = (8,6))
    cmap = sns.color_palette("Accent", 5)

    cmap[0] = (0,0,0) # Zwart
    cmap[1] = (1,0,0) # Rood
    cmap[2] = (255/255,150/255,8/255) # Blauw
    cmap[3] = (245/255,245/255,245/255)    #Grijs
    cmap[4] = (0,0.7,0) #Groen
    ax = sns.heatmap(df[performance_measures], cmap = cmap, linewidths=.005, yticklabels = 10)
    colorbar = ax.collections[0].colorbar 
    colorbar.set_ticks([colorbar.vmin + 4 / 5 * (0.5 + i) for i in range(5)])
    colorbar.set_ticklabels(["N.D.","Min","Min = Max","Not-optimal","Max"])
    plt.title(r"Optimal Expectations Measures and Metrics with P:" + str(P) + " and N:" + str(N))
    plt.ylabel(r"$\theta^*$")
    plt.xlabel("Measure/ Metric")
    plt.xticks(rotation=70)
    plt.show()

def determine_optima(metrics_list, M, stepsize):
    results = []
    for measure in metrics_list:
        for p in range(0, M + 1, stepsize):
            y_true = [1] * int(p * 0.01 * M) + [0] * int((100 - p) * 0.01 * M)
            try:
                outcome = optimized_basic_baseline(y_true, measure)
            except:
                outcome = {'Max Expected Value': np.nan, 'Min Expected Value': np.nan}
            results.append([measure, p, M, outcome["Min Expected Value"], outcome["Max Expected Value"]])
    df = pd.DataFrame(results, columns = ["Metric", "P", "M", "E_min", "E_max"])
    return df
           
#%%

df = get_baseline()

plot_metrics_over_theta(df, BASE_MEASURES , "Amount")
plot_metrics_over_theta(df, BASE_METRICS , "Score")
plot_metrics_over_theta(df, HIGHER_ORDER_METRICS, "Score")

#%%

df_pivot = pd.pivot_table(data = df.round({"Theta":2}), values = "Score", 
                          columns = "Metric", index = "Theta").round(5).abs().sort_index(ascending=False)

plot_heatmap(df_pivot, BASE_MEASURES, P, N)
plot_heatmap(df_pivot, BASE_METRICS, P, N)
plot_heatmap(df_pivot, HIGHER_ORDER_METRICS, P, N)

#%%

df_extrema = translate_scores_to_extrema(df_pivot)
plot_heatmap_extrema(df_extrema)

#%%
M = 100
df = determine_optima([x for x in HIGHER_ORDER_METRICS if x != "G2"], M, 1)

plt.figure(figsize = (8,8))
sns.lineplot(x = "P", y = "E_max", hue = "Metric", data = df, alpha=0.7 )
plt.title(r"Expectation Base Measures over different $\theta$ with M:" + str(M))
plt.ylabel("Maximum Expected Score")
plt.show()

# To do:
# BM = Y
# COHEN = KAPPA
# G1 = FM


