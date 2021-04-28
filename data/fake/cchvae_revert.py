import pandas as pd
from pathlib import Path

dataDir = Path(__file__).parent.absolute()

fake_c = pd.read_csv(f"{dataDir}/fake_counterfactuals.csv", header=None)
fake_d_c = pd.read_csv(f"{dataDir}/fake_delta_counterfactuals.csv", header=None)

p_c = fake_c.iloc[1:,:]
p_c = p_c / 100
pos = p_c[(p_c[0]>= 0.5) & (p_c[1]>= 0.5) & (p_c[2]>= 0.5) & (p_c[3]>= 0.5)].count()
neg = p_c[(p_c[0]<= -0.5) & (p_c[1]<= -0.5) & (p_c[2]<= -0.5) & (p_c[3]<= -0.5)].count()
pos_pos = p_c[(p_c[0]<= 0.5) & (p_c[1]<= 0.5) & (p_c[2]<= 0.5) & (p_c[3]<= 0.5)].count()
neg_neg = p_c[(p_c[0]>= -0.5) & (p_c[1]>= -0.5) & (p_c[2]>= -0.5) & (p_c[3]>= -0.5)].count()
print(pos)
print(neg)
print(pos_pos)
print(neg_neg)
print(len(p_c))

# pos = 0
# neg = 0
# for v in p_c[0]:
#     if v >= 0.5:
#         # print("if", v)
#         pos += 1
#         pass
#     elif v <= -0.5:
#         # print("elif", v)
#         neg += 1
#         pass
#     else:
#         # print("else", v)
#         pass

# print(pos)
# print(neg)
