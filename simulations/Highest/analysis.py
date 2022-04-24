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

data_in = '/Users/biocomplexity/Projects/SocioCognitiveModeling/Metaculus_CogModeling/simulations/Preprocessing'
infile_name = 'sim_out_highest'

df_s_freqs = pd.read_csv(f'{data_in}/Final_Cleaned_High_R1_R3_S_freqs.csv')
s_freq_threshold = 15
s_S = df_s_freqs.loc[df_s_freqs.decision_date>s_freq_threshold,'user_id']
s_list = s_S.to_list()
group_list = ['all_s'] + s_list

'''PUT ALL DATA FOR EACH GROUP IN LIST'''
catch_groups = []
for i in group_list:
    '''ANALYZE All Ss'''
    infile_group = i

    '''GRAB DATA'''
    S_all = pd.read_pickle(f'S_all_{infile_name}_{infile_group}.csv')

    
'''PLOT MESSY'''
counter=0
for i in catch_groups: 
    if counter==0:
        plt.scatter(i.index,i.p_dur)
        i.prior.rolling(14).mean().plot(label='optimal')
        i.hum.rolling(14).mean().plot(label='human')
        i.p_dur.rolling(14).mean().plot(label='raw')
    else:
        i.prior.rolling(14).mean().plot(label='optimal')
        i.hum.rolling(14).mean().plot(label='human')
        i.p_dur.rolling(14).mean().plot(label='raw')
    
    #plt.legend()

    counter =+ 1


'''CLEAN: GNEERATES PNGS FOR EACH GROUP
DOESNT QUITE WORK'''
counter = 0 #FOR GROUP NAME in group_list
for i in catch_groups: 
        print(counter)
        plt.scatter(i.index,i.p_dur)
        i.prior.rolling(14).mean().plot(label='optimal')
        i.hum.rolling(14).mean().plot(label='human')
        i.p_dur.rolling(14).mean().plot(label='raw')
        plt.savefig(f'{infile_name}_{group_list[counter]}.png',dpi=200)
        counter += 1
    
    
    

'''DEV HERE WITHOUT LOOP'''

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
catch.append(S_all)


'''SUBJECT X'''
'''ANALYZE All Ss'''
infile_group = 112076

'''GRAB DATA'''
S_all = pd.read_pickle(f'S_all_{infile_name}_{infile_group}.csv')
#FOR ANALYSIS
S_all.plot()
plt.scatter(S_all.index,S_all.hum)

S_all.prior.rolling(14).mean().plot(label='optimal')
S_all.hum.rolling(14).mean().plot(label='human')
S_all.p_dur.rolling(14).mean().plot(label='raw')
plt.legend()

#EOF