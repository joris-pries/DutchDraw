import matplotlib.pyplot as plt
from scipy.stats import hypergeom

def expected_TS(P, M, theta):
    rounded_m_theta = round(M * theta)
    lb = max(0, rounded_m_theta - M + P)
    ub = min(P, rounded_m_theta)
    TP_rv = hypergeom(M=M, n=P, N=rounded_m_theta)
    exp = 0
    for k in range(lb, ub+1):
        exp = exp + (k / (P + rounded_m_theta - k)) * TP_rv.pmf(k)
    return exp

P = 2
M = 10

expectations = []
thetas = []
for i in range(101):
    theta = i / 100
    thetas.append(theta)
    expectations.append(expected_TS(P, M, theta))


plt.figure(figsize = (10,10))
plt.plot(thetas,expectations)
plt.axhline(y = P/M, color="red")
plt.xlabel("Theta")
plt.ylabel("E[TS]")
plt.title("M: " + str(M) + ", P: " + str(P))
plt.show()