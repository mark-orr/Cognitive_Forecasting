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







'''QUICK TEMP CHECK ON INDIVIDUALS FOR LOWER LEFT PLOT'''
'''PANELS'''
'''ALL GPs on ONE PLOT'''
fig_ax_dim_x=2
fig_ax_dim_y=4
fig, axes = plt.subplots(fig_ax_dim_y,fig_ax_dim_x,figsize=(8,8),sharex=True,sharey=True)
leg_x = .92
leg_y = 1
#PANEL 0,0
gp_ordering = np.array([8,7,6,5,4,3,2,1])
gp_counter = 0
for k in range(0,fig_ax_dim_y):
    for j in range(0,fig_ax_dim_x):
        print(i,j)
        gp_no = group_list[gp_ordering[gp_counter]]
        i = catch_groups[gp_ordering[gp_counter]]
        '''aVE OVER DAY'''
        grouped = i.groupby(level=0)
        axes[k,j].scatter(i.p_dur,i.hum-i.prior,color='black',label='horizon by t_predicted - prior',marker='^',s=50,alpha=0.25)
        axes[k,j].text(60,10,'P'+str(gp_counter+1),fontsize=10)
        axes[k,j].axvline(x=5,c='black',dashes=(2,2,2,2),linewidth=2,alpha=0.5)
        gp_counter += 1
plt.subplots_adjust(wspace=.1,hspace=0.1)
plt.savefig(f'Tmp1.png', dpi=300, transparent=False, bbox_inches='tight')

fig_ax_dim_x=2
fig_ax_dim_y=3
fig, axes = plt.subplots(fig_ax_dim_y,fig_ax_dim_x,figsize=(8,8),sharex=True,sharey=True)
leg_x = .92
leg_y = 1
#PANEL 0,0
gp_ordering = np.array([8,7,6,5,4,3])
gp_counter = 0
for k in range(0,fig_ax_dim_y):
    for j in range(0,fig_ax_dim_x):
        print(i,j)
        gp_no = group_list[gp_ordering[gp_counter]]
        i = catch_groups[gp_ordering[gp_counter]]
        i['hum_minus_prior'] = i.hum - i.prior
        '''aVE OVER DAY'''
        grouped = i.groupby(level=0)
        axes[k,j].scatter(i.index,i.hum_minus_prior,color='black',label='t_predicted - prior',marker='^',s=50,alpha=0.25)
        #axes[k,j].plot(grouped.hum_minus_prior.mean().rolling(4).mean(),color='black')
        axes[k,j].plot(i.hum_minus_prior,color='black')
        #axes[k,j].text(60,10,'P'+str(gp_counter+1),fontsize=10)
        #axes[k,j].axvline(x=5,c='black',dashes=(2,2,2,2),linewidth=2,alpha=0.5)
        gp_counter += 1
plt.subplots_adjust(wspace=.1,hspace=0.1)
plt.savefig(f'Tmp2.png', dpi=300, transparent=False, bbox_inches='tight')
'''END QUICK CHECK'''

'''WHICH OF THESE TWO IS MORE INFORMATIVE?'''




'''THIS WORKS'''
for k in range(0,len(gp_ordering)):
    gp_counter = k
    
    gp_no = group_list[gp_ordering[gp_counter]]
    i = catch_groups[gp_ordering[gp_counter]]
    i['hum_minus_prior'] = i.hum - i.prior
    '''aVE OVER DAY'''
    grouped = i.groupby(level=0)
    plt.scatter(i.index,i.hum_minus_prior,color='black',label='S' + str(k+1),marker=k,s=50,alpha=0.25)
    #axes[k,j].plot(grouped.hum_minus_prior.mean().rolling(4).mean(),color='black')
    plt.plot(i.hum_minus_prior,color='black',dashes=(2,2,2,2),alpha=.5,linewidth=0.5)

plt.legend()


'''THIS WORKS'''
for k in range(0,len(gp_ordering)):
    gp_counter = k
    
    gp_no = group_list[gp_ordering[gp_counter]]
    i = catch_groups[gp_ordering[gp_counter]]
    i['hum_minus_prior'] = i.hum - i.prior
    '''aVE OVER DAY'''
    grouped = i.groupby(level=0)
    plt.scatter(i.p_dur,i.hum_minus_prior,color='black',label='S' + str(k+1),marker=k,s=50,alpha=0.25)
    
plt.axvline(x=5,c='black',dashes=(2,2,2,2),linewidth=2,alpha=0.5)
plt.legend()


'''PANELS'''
'''ALL GPs on ONE PLOT'''
fig_ax_dim_x=2
fig_ax_dim_y=4
fig, axes = plt.subplots(fig_ax_dim_y,fig_ax_dim_x,figsize=(8,8),sharex=True,sharey=True)
leg_x = .92
leg_y = 1
#PANEL 0,0
gp_ordering = np.array([8,7,6,5,4,3,2,1])
gp_counter = 0
for k in range(0,fig_ax_dim_y):
    for j in range(0,fig_ax_dim_x):
        print(i,j)
        gp_no = group_list[gp_ordering[gp_counter]]
        i = catch_groups[gp_ordering[gp_counter]]
        '''aVE OVER DAY'''
        grouped = i.groupby(level=0)
        axes[k,j].scatter(i.index, i.p_dur,color='black',label='human horizon',marker='+',s=150,alpha=0.25)
        axes[k,j].plot(grouped.prior.mean().rolling(4).mean(),color='black',label='rational prior',dashes=(0,2,2,2))
        axes[k,j].plot(grouped.hum.mean().rolling(4).mean(),color='black',label='human t_predicted')
        axes[k,j].plot(grouped.p_dur.mean().rolling(4).mean(),color='black',label='human horizon',dashes=(0,0,2,2))
        axes[k,j].scatter(gt_S.index,gt_S,color='black',label='ground truth horizon',marker='o',s=10,alpha=0.45)
        axes[k,j].legend(bbox_to_anchor=(leg_x,leg_y), loc=1, frameon=False, fontsize=5)
        axes[k,j].set_ylim(0, 86)
        axes[k,j].axvline(x=datetime.strptime('2021-12-03','%Y-%m-%d'),c='black',dashes=(6,6,6,6),linewidth=1)
        axes[k,j].axvline(x=datetime.strptime('2021-12-24','%Y-%m-%d'),c='black',dashes=(6,6,6,6),linewidth=1)
        axes[k,j].axvline(x=datetime.strptime('2022-01-14','%Y-%m-%d'),c='black',dashes=(6,6,6,6),linewidth=1)
        axes[k,j].text(datetime.strptime('2021-12-10','%Y-%m-%d'),75,'P'+str(gp_counter+1),fontsize=10)
        gp_counter += 1
#LABELS
axes[3,0].set_xlabel('Date', labelpad=10,size=10)
axes[3,0].tick_params(axis='x', labelsize=5)
axes[3,1].set_xlabel('Date', labelpad=10,size=10)
axes[3,1].tick_params(axis='x', labelsize=5)

axes[0,0].set_ylabel('Days', labelpad=10,size=10)
axes[1,0].set_ylabel('Days', labelpad=10,size=10)
axes[2,0].set_ylabel('Days', labelpad=10,size=10)
axes[3,0].set_ylabel('Days', labelpad=10,size=10)
plt.subplots_adjust(wspace=.1,hspace=0.1)
plt.savefig(f'Single_Subjects.png', dpi=300, transparent=False, bbox_inches='tight')




#EOF