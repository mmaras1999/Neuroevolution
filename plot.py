# script for plotting performance plots for CMA-ES, MA-ES and LM-MA-ES
import numpy as np
import pickle
import matplotlib.pyplot as plt
from lib.utilities import load_obj

model = load_obj(550, 'models/cmaes_pong_v18')
scores = [] 

for scr in model.bestgens:
    if type(scr) == tuple:
        scores.append(-scr[1]) 
    else:
        scores.append(-scr)

model.genmeans = -np.array(model.genmeans)

plt.plot(scores, label='best')
plt.plot(model.genmeans, label='mean')

aver = 20
plt.plot(range(aver-1, len(scores)), np.convolve(scores, np.ones(aver) / aver, mode='valid'), label=f'moving best avg {aver}')
plt.plot(range(aver-1, len(model.genmeans)), np.convolve(model.genmeans, np.ones(aver) / aver, mode='valid'), label=f'moving mean avg {aver}')
plt.legend()
# plt.savefig('plots/2048_lm_ma_es.png')
plt.show()
