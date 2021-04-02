# %%
M = 5000
prec_1 = 0.35
recall_1 = 0.24
prec_2 = 0.35
recall_2 = 0.30

for P in range(M + 1):
    TP_1 = P * recall_1
    FP_1 = ((1 - prec_1) / prec_1) * TP_1

    TP_2 = P * recall_2
    FP_2 = ((1 - prec_2) / prec_2) * TP_2


# %%
