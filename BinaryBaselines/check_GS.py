import matplotlib.pyplot as plt
from scipy.stats import hypergeom

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

P = 2
M = 10
N = M - P
expectations = []
thetas = []
upper_limit = []
for i in range(M+1):
    theta = i / M
    upper_limit.append((theta * (1 - theta) * M / (M - 1)) ** 0.5)
    thetas.append(theta)
    expectations.append(expected_G2(P, M, theta))


plt.figure(figsize = (10,10))
plt.plot(thetas,expectations, label = r"$E[G_\theta^2]$")
plt.plot(thetas, upper_limit, label = r"$\sqrt{(E[(G_\theta^2)^2])}$")
plt.xlabel(r"$\theta$")
plt.ylabel(r"$E[G_\theta^2]$")
plt.axhline(y = ((P * N) ** 0.5) / M, color = "yellow", label = r"$E[G_{\frac{N}{M}}^2]$")
plt.axvline(x = N / M, color ="red", label = r"$\frac{N}{M}}$")
plt.title("M: " + str(M) + ", P: " + str(P))
plt.legend()
plt.show()