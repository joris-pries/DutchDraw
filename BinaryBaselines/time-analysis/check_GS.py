import matplotlib.pyplot as plt
from scipy.stats import hypergeom
import numpy as np

def expected_G2(P, M, theta):
    N = M - P
    rounded_m_theta = round(M * theta)
    lb = max(0, rounded_m_theta - M + P)
    ub = min(P, rounded_m_theta)
    TP_rv = hypergeom(M=M, n=P, N=rounded_m_theta)
    exp = 0
    for k in range(lb, ub+1):
        exp = exp + ((k * (N - rounded_m_theta + k) / (P*N))**0.5 )* TP_rv.pmf(k)
    return exp

def shuffle_simulate_scores(d, P, N, s):
    n = N + P
    tp = hypergeom.rvs(n, d, P, size=s)
    scores = np.array((tp * (N - d + tp) / (P * N))**0.5)
    return scores

P = 8
M = 10
N = M - P
expectations = []
thetas = []
upper_limit = []
upper_limit_2 = []
for i in range(100+1):
    theta = i / 100
    theta_star = round(M * theta) / M
    upper_limit.append((theta_star * (1 - theta_star) * M / (M - 1)) ** 0.5)
    thetas.append(theta)
    expectations.append(expected_G2(P, M, theta))

plt.figure(figsize = (10,10))
plt.plot(thetas,expectations, label = r"$E[G_\theta^{(2)}]$")
plt.plot(thetas, upper_limit, label = r"$\sqrt{(E[(G_\theta^{(2)})^2])}$")
plt.xlabel(r"$\theta$")
plt.ylabel(r"$E[G_\theta^{(2)}]$")
plt.axhline(y = ((P * N) ** 0.5) / M, color = "yellow", label = r"$E[G_{\frac{N}{M}}^{(2)}]$")
plt.axvline(x = N / M, color ="red", label = r"$\frac{N}{M}}$")
plt.title("M: " + str(M) + ", P: " + str(P))
plt.legend()
plt.show()

P = 100
N = 1000
M = N + P
theta = 0.1
s = 10000000
d = round(M * theta)
theta_star = round(M * theta) / M


scores = shuffle_simulate_scores(d, P, N, s)
print(np.mean(scores) - expected_G2(P, M, theta))
var_ = (theta_star * (1 - theta_star) * M / (M - 1)) - expected_G2(P, M, theta)**2 
print(np.var(scores) - var_)







