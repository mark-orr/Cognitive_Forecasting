import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import pickle

from imp import reload
import sys
sys.path.insert(1,'/Users/biocomplexity/Projects/SocioCognitiveModeling/Metaculus_CogModeling/bayes_pack_testing')
import bayes
import b_fit

reload(bayes)

infile_name = 'sim_out_highest'


'''ANALYZE All Ss'''
infile_group = 'all_S'

'''GRAB DATA'''
S_all = pd.read_pickle(f'S_all_{infile_name}_{infile_group}.csv')
#FOR ANALYSIS
S_all.plot()
plt.scatter(S_all.index,S_all.hum)

S_all.prior.rolling(14).mean().plot(label='optimal')
S_all.hum.rolling(14).mean().plot(label='human')
S_all.p_dur.rolling(14).mean().plot(label='raw')
plt.legend()

S_all.prior.plot()
#EOF