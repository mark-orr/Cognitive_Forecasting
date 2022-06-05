import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import pickle

import matplotlib.font_manager as fm   
from pylab import cm
import matplotlib as mpl

from imp import reload
import sys
sys.path.insert(1,'/Users/biocomplexity/Projects/SocioCognitiveModeling/Metaculus_CogModeling/bayes_pack_testing')
import bayes
import b_fit

reload(bayes)

data_in = '/Users/biocomplexity/Projects/SocioCognitiveModeling/Metaculus_CogModeling/simulations/Preprocessing'
cases_data_in = '/Users/biocomplexity/Projects/SocioCognitiveModeling/Metaculus_CogModeling/simulations/InputData'
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
    
    catch_groups.append(S_all)


'''THIS CODE PUT HERE JUST TO GENERATE INDEX FOR GT'''    
gp_no = group_list[0]
i = catch_groups[0]
'''aVE OVER DAY'''
grouped = i.groupby(level=0)

'''ADDING GT'''
tmp1 = np.arange(1,45)
tmp2 = tmp1[::-1]
tmp2 = np.append(tmp2,0)
tmp2 = np.append(tmp2,-1)
gt_S = pd.Series(tmp2)
gt_S.index = grouped.prior.mean().rolling(4).mean().index #NEED TO RUN BELOW FIRST


marker_list = ['x','+','o','v','^','*']

'''THIS WORKS'''
for k in range(0,len(gp_ordering)):
    gp_counter = k
    
    gp_no = group_list[gp_ordering[gp_counter]]
    i = catch_groups[gp_ordering[gp_counter]]
    i['hum_minus_prior'] = i.hum - i.prior
    '''aVE OVER DAY'''
    grouped = i.groupby(level=0)
    plt.scatter(i.index,i.hum_minus_prior,color='black',label='P' + str(k+1),marker=marker_list[k],s=50,alpha=0.15)
    #axes[k,j].plot(grouped.hum_minus_prior.mean().rolling(4).mean(),color='black')
    plt.plot(i.hum_minus_prior,color='black',dashes=(2,2,2,2),alpha=.5,linewidth=0.5)

plt.legend(frameon=False, fontsize=11)
plt.xlabel('Date', labelpad=5,size=15)
plt.tick_params(axis='x', labelsize=8)
plt.ylabel('human t_pred - rational prior (Days)', labelpad=10,size=15)
plt.tick_params(axis='y', labelsize=15)
plt.savefig('test_new.png',dpi=300,transparent=False, bbox_inches='tight')


'''THIS WORKS'''
for k in range(0,len(gp_ordering)):
    gp_counter = k
    
    gp_no = group_list[gp_ordering[gp_counter]]
    i = catch_groups[gp_ordering[gp_counter]]
    i['hum_minus_prior'] = i.hum - i.prior
    '''aVE OVER DAY'''
    grouped = i.groupby(level=0)
    plt.scatter(i.p_dur,i.hum_minus_prior,color='black',label='S' + str(k+1),marker=marker_list[k],s=50,alpha=0.25)
    
plt.axvline(x=5,c='black',dashes=(2,2,2,2),linewidth=2,alpha=0.5)
plt.legend(frameon=False, fontsize=11)
plt.xlabel('human horizon (Days)', labelpad=5,size=15)
plt.tick_params(axis='x', labelsize=15)
plt.ylabel('human t_pred - rational prior (Days)', labelpad=10,size=15)
plt.tick_params(axis='y', labelsize=15)
plt.savefig('test_new_2.png',dpi=300,transparent=False, bbox_inches='tight')


#EOF