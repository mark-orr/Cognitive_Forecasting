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
    

    
'''FANCIER METHOD'''
mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
font_names = [f.name for f in fm.fontManager.ttflist]
print(font_names)


'''ALL GPs on ONE PLOT'''
plt.style.use('default')

fig = plt.figure(figsize=(7, 5))

ax = fig.add_axes([0, 0, 2, 2])
ax.set_title(f'Human Judgments and Estimated Priors',size=20)
ax.set_xlabel('Date', labelpad=10,size=20)
ax.set_ylabel('Days', labelpad=10,size=20)

ax.set_ylim(-10, 90)
ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

gp_no = group_list[0]
i = catch_groups[0]
'''aVE OVER DAY'''
grouped = i.groupby(level=0)
ax.scatter(i.index, i.p_dur,color='black',label='human horizon',marker='+',s=150,alpha=0.25)
ax.plot(grouped.prior.mean().rolling(4).mean(),color='black',label='prior',dashes=(0,2,2,2))
ax.plot(grouped.hum.mean().rolling(4).mean(),color='black',label='human t_total')
ax.plot(grouped.p_dur.mean().rolling(4).mean(),color='black',label='human horizon',dashes=(0,0,2,2))
#ax.plot(df_test_mean_forward.delta,color='black',label='human t_total delta',dashes=(6,6,6,6),alpha=0.3)

ax.axvline(x=datetime.strptime('2021-12-24','%Y-%m-%d'),c='black',dashes=(6,6,6,6),linewidth=1)
ax.axvline(x=datetime.strptime('2022-01-14','%Y-%m-%d'),c='black',dashes=(6,6,6,6),linewidth=1)
ax.axhline(y=42,c='black',dashes=(2,2,2,2),linewidth=1,alpha=0.2)
ax.axhline(y=0,c='black',dashes=(2,2,2,2),linewidth=2,alpha=0.3)
ax.legend(bbox_to_anchor=(.85, .85), loc=1, frameon=False, fontsize=20)
#plt.savefig(f'Good_{gp_no}.png', dpi=300, transparent=False, bbox_inches='tight')
plt.show()





'''
NEXT MAKE DELTA PLOT
AND ADD EPI CURVE DELTA PLOT
THEN MAKE PANEL FOR ALL Ss
'''
#ERROR PER GROUPTING
grouped.hum.std()/grouped.hum.count()
test_mean = grouped.hum.mean().rolling(4).mean()

fwd_value = 5
test_mean_forward = test_mean.copy()
test_mean_forward = test_mean_forward.reset_index(drop=True)
test_mean_forward = test_mean_forward[:-fwd_value].reset_index(drop=True)
test_mean_forward = pd.concat([pd.Series(np.full(fwd_value,fill_value=np.NaN)),test_mean_forward],axis=0)
test_mean_forward.index = test_mean.index
test_mean_forward.name = 'forward'

df_test_mean_forward = pd.concat([test_mean,test_mean_forward],axis=1)
df_test_mean_forward['delta'] = df_test_mean_forward.hum - df_test_mean_forward.forward
plt.plot(df_test_mean_forward.delta)
plt.plot(df_test_mean_forward.delta/df_test_mean_forward.delta.max())
df_test_mean_forward['delta_norm']=df_test_mean_forward.delta/df_test_mean_forward.delta.max()

'''ADD EPI'''
'''DATA FROM VIRGINIA'''
#SRINI RECOMMEND:
'''/project/biocomplexity/COVID-19_commons/data/VDH_public/VDH-COVID-19-PublicUseDataset-Cases.csv'''

vdh = pd.read_csv(f'{cases_data_in}/VDH-COVID-19-PublicUseDataset-Cases.csv',parse_dates=['report_date'])

vdh = vdh.groupby(['report_date'])['total_cases'].sum().diff().rolling(window=7).mean().round()
vdh_use = vdh['11-30-2021':'01-14-2022'].copy()
vdh_use.name = 'use'

fwd_value = 5
vdh_use_forward = vdh_use.copy()
vdh_use_forward = vdh_use_forward.reset_index(drop=True)
vdh_use_forward = vdh_use_forward[:-fwd_value].reset_index(drop=True)
vdh_use_forward = pd.concat([pd.Series(np.full(fwd_value,fill_value=np.NaN)),vdh_use_forward],axis=0)
vdh_use_forward.index = vdh_use.index
vdh_use_forward.name = 'forward'

df_vdh_use_forward = pd.concat([vdh_use,vdh_use_forward],axis=1)
df_vdh_use_forward['delta'] = df_vdh_use_forward.use - df_vdh_use_forward.forward
plt.plot(df_vdh_use_forward.delta)
plt.plot(df_vdh_use_forward.delta/df_vdh_use_forward.delta.max())
df_vdh_use_forward['delta_norm']=df_vdh_use_forward.delta/df_vdh_use_forward.delta.max()

'''PLOT TOGETHER'''
plt.style.use('default')

fig = plt.figure(figsize=(7, 5))

ax = fig.add_axes([0, 0, 2, 2])
ax.set_title(f'Delta Comparison',size=20)
ax.set_xlabel('Date', labelpad=10,size=20)
ax.set_ylabel('Normed Delta', labelpad=10,size=20)

ax.set_ylim(-1.8, 1.8)
ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


ax.plot(df_test_mean_forward.delta_norm,color='black',label='human t_total delta',alpha=0.7)
ax.plot(df_vdh_use_forward.delta_norm,color='black',label='cases',dashes=(2,2,2,2),alpha=0.7,linewidth=1.3)

ax.axvline(x=datetime.strptime('2021-12-24','%Y-%m-%d'),c='black',dashes=(6,6,6,6),linewidth=1)
ax.axvline(x=datetime.strptime('2022-01-14','%Y-%m-%d'),c='black',dashes=(6,6,6,6),linewidth=1)
ax.axhline(y=0,c='black',dashes=(2,2,2,2),linewidth=2,alpha=0.3)
ax.legend(bbox_to_anchor=(.93,1), loc=1, frameon=False, fontsize=20)
plt.savefig(f'Good_Delta{gp_no}.png', dpi=300, transparent=False, bbox_inches='tight')
plt.show()


'''END ALL S PLOTTING'''







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
        axes[k,j].plot(grouped.prior.mean().rolling(4).mean(),color='black',label='prior',dashes=(0,2,2,2))
        axes[k,j].plot(grouped.hum.mean().rolling(4).mean(),color='black',label='human t_total')
        axes[k,j].plot(grouped.p_dur.mean().rolling(4).mean(),color='black',label='human horizon',dashes=(0,0,2,2))
        axes[k,j].legend(bbox_to_anchor=(leg_x,leg_y), loc=1, frameon=False, fontsize=5)
        axes[k,j].set_ylim(0, 86)
        axes[k,j].axvline(x=datetime.strptime('2021-12-24','%Y-%m-%d'),c='black',dashes=(6,6,6,6),linewidth=1)
        axes[k,j].axvline(x=datetime.strptime('2022-01-14','%Y-%m-%d'),c='black',dashes=(6,6,6,6),linewidth=1)
        axes[k,j].text(datetime.strptime('2021-12-01','%Y-%m-%d'),75,gp_no,fontsize=10)
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



'''PANELS FOR DELTA'''
fig_ax_dim_x=2
fig_ax_dim_y=4
fig, axes = plt.subplots(fig_ax_dim_y,fig_ax_dim_x,figsize=(8,8),sharex=True,sharey=True)
leg_x = .92
leg_y = 1
#PANEL 0,0
gp_ordering = [8,7,6,5,4,3,2,1]
gp_counter = 0
for k in range(0,fig_ax_dim_y):
    for j in range(0,fig_ax_dim_x):
        print(k,j,': is k,j')
        print(gp_counter,': is GP COUNTER')
        gp_no = group_list[gp_ordering[gp_counter]]
        i = catch_groups[gp_ordering[gp_counter]]
        '''aVE OVER DAY'''
        grouped = i.groupby(level=0)
        
        '''ADD IN THE DELTA COMPUTE'''
        test_mean = grouped.hum.mean().rolling(4).mean()

        fwd_value = 5
        test_mean_forward = test_mean.copy()
        test_mean_forward = test_mean_forward.reset_index(drop=True)
        test_mean_forward = test_mean_forward[:-fwd_value].reset_index(drop=True)
        test_mean_forward = pd.concat([pd.Series(np.full(fwd_value,fill_value=np.NaN)),test_mean_forward],axis=0)
        test_mean_forward.index = test_mean.index
        test_mean_forward.name = 'forward'

        df_test_mean_forward = pd.concat([test_mean,test_mean_forward],axis=1)
        df_test_mean_forward['delta'] = df_test_mean_forward.hum - df_test_mean_forward.forward
        df_test_mean_forward['delta_norm']=df_test_mean_forward.delta/df_test_mean_forward.delta.max()
        
        
        axes[k,j].plot(df_test_mean_forward.delta_norm,color='black',label='human t_total delta',alpha=0.7)
        axes[k,j].plot(df_vdh_use_forward.delta_norm,color='black',label='cases',dashes=(2,2,2,2),alpha=0.7,linewidth=1.3)
        
        
        axes[k,j].legend(bbox_to_anchor=(leg_x,leg_y), loc=1, frameon=False, fontsize=5)
        axes[k,j].set_ylim(-1.8, 1.8)
        axes[k,j].axvline(x=datetime.strptime('2021-12-24','%Y-%m-%d'),c='black',dashes=(6,6,6,6),linewidth=1)
        axes[k,j].axvline(x=datetime.strptime('2022-01-14','%Y-%m-%d'),c='black',dashes=(6,6,6,6),linewidth=1)
        axes[k,j].text(datetime.strptime('2021-12-06','%Y-%m-%d'),1.5,gp_no,fontsize=10)
        gp_counter += 1
#LABELS
axes[3,0].set_xlabel('Date', labelpad=10,size=10)
axes[3,0].tick_params(axis='x', labelsize=5)
axes[3,1].set_xlabel('Date', labelpad=10,size=10)
axes[3,1].tick_params(axis='x', labelsize=5)

axes[0,0].set_ylabel('Normed Delta', labelpad=10,size=10)
axes[1,0].set_ylabel('Normed Delta', labelpad=10,size=10)
axes[2,0].set_ylabel('Normed Delta', labelpad=10,size=10)
axes[3,0].set_ylabel('Normed Delta', labelpad=10,size=10)
plt.subplots_adjust(wspace=.1,hspace=0.1)

plt.savefig(f'Single_Subjects_Delta.png', dpi=300, transparent=False, bbox_inches='tight')




#EOF