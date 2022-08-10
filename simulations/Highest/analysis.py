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
    print(infile_group)

    '''GRAB DATA'''
    S_all = pd.read_pickle(f'S_all_{infile_name}_{infile_group}.csv')
    
    catch_groups.append(S_all)



'''ADDITION OF GT'''
'''THIS grouped CODE PUT HERE JUST TO GENERATE INDEX FOR GT'''    
gp_no = group_list[0]
i = catch_groups[0]
'''aVE OVER DAY'''
grouped = i.groupby(level=0)

tmp1 = np.arange(1,45)
tmp2 = tmp1[::-1]
tmp2 = np.append(tmp2,0)
tmp2 = np.append(tmp2,-1)
gt_S = pd.Series(tmp2)
gt_S.index = grouped.prior.mean().rolling(4).mean().index #NEED TO RUN BELOW FIRST




'''ALL GPs on ONE PLOT'''
plt.style.use('default')

fig = plt.figure(figsize=(7, 5))

ax = fig.add_axes([0, 0, 2, 2])
#ax.set_title(f'Human Judgments and Estimated Priors',size=20)
ax.set_xlabel('Date', labelpad=10,size=20)
ax.set_ylabel('Days', labelpad=10,size=20)

ax.set_ylim(-10, 90)
ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
#ax.spines['right'].set_visible(False)
#ax.spines['top'].set_visible(False)

gp_no = group_list[0]
i = catch_groups[0]
'''aVE OVER DAY'''
grouped = i.groupby(level=0)
ax.scatter(i.index, i.p_dur,color='black',label='human horizon',marker='+',s=150,alpha=0.25)
ax.plot(grouped.prior.mean().rolling(4).mean(),color='black',label='rational prior',dashes=(0,2,2,2))
ax.plot(grouped.hum.mean().rolling(4).mean(),color='black',label='human t_predicted')
ax.plot(grouped.p_dur.mean().rolling(4).mean(),color='black',label='human horizon',dashes=(0,0,2,2))
ax.plot(gt_S,color='black',label='ground truth horizon',dashes=(3,3,10,5))
ax.scatter(gt_S.index,gt_S,color='black',label='ground truth horizon',marker='p',s=150,alpha=0.45)
#ax.plot(df_test_mean_forward.delta,color='black',label='human t_total delta',dashes=(6,6,6,6),alpha=0.3)

#ax.axvline(x=datetime.strptime('2021-12-03','%Y-%m-%d'),c='black',dashes=(2,2,2,2),linewidth=1,alpha=0.3)
#ax.axvline(x=datetime.strptime('2021-12-24','%Y-%m-%d'),c='black',dashes=(2,2,2,2),linewidth=1,alpha=0.3)
#ax.axvline(x=datetime.strptime('2022-01-14','%Y-%m-%d'),c='black',dashes=(2,2,2,2),linewidth=1,alpha=0.3)
#ax.axhline(y=42,c='black',dashes=(2,2,2,2),linewidth=1,alpha=0.2)
ax.axhline(y=0,c='black',dashes=(2,2,2,2),linewidth=1,alpha=0.3)
ax.legend(bbox_to_anchor=(.90, .95), loc=1, frameon=False, fontsize=20)
#plt.savefig(f'Good_{gp_no}.png', dpi=300, transparent=False, bbox_inches='tight')
plt.show()


'''
SOME STATISTICS ON THE PLOT
FOR THE MANUSCRIPT
AND SUMMARY PLOT
'''
#SET UP DATA
i = catch_groups[0].copy()
i['hum_minus_prior'] = i.hum - i.prior

grouped = i.groupby(level=0)
#MEAN OF PRIOR
grouped.prior.mean().mean()
grouped.prior.mean().std()
#MEAN OF human t_predicted
grouped.hum.mean().mean()
grouped.hum.mean().std()

'''
SUMMARY GRAPH FOR ALL Ss FIGURE
'''
fig_ax_dim_x=2
fig_ax_dim_y=2
fig, axes = plt.subplots(fig_ax_dim_y,fig_ax_dim_x,figsize=(16,8),sharex=False,sharey=False)
leg_x = .92
leg_y = 1

axes[0,1].plot(grouped.hum.mean().rolling(4).mean(),color='black',label='t_predicted')
axes[0,1].plot(grouped.prior.mean().rolling(4).mean(),color='black',label='prior',dashes=(4,2,4,2),linewidth=2)
axes[0,1].scatter(i.index,i.hum,color='black',label='t_predicted',marker='o',s=50,alpha=0.25)
axes[0,1].scatter(i.index,i.prior,color='black',label='prior',marker='2',s=50,alpha=0.25)
axes[0,1].legend(bbox_to_anchor=(.99, .99), loc=1, frameon=False, fontsize=9)


axes[1,1].scatter(i.index,i.hum_minus_prior,color='black',label='t_predicted - prior',marker='^',s=50,alpha=0.25)
axes[1,1].plot(grouped.hum_minus_prior.mean().rolling(4).mean(),color='black')
axes[1,0].scatter(i.p_dur,i.hum-i.prior,color='black',label='horizon by t_predicted - prior',marker='^',s=50,alpha=0.25)
axes[0,0].scatter(i.hum,i.prior,color='black',label='prior-t_predicted',marker='^',s=75,alpha=0.25)

axes[1,0].axvline(x=5,c='black',dashes=(2,2,2,2),linewidth=2,alpha=0.5)

#LABELS
axes[0,1].set_xlabel('Date', labelpad=5,size=15)
axes[0,1].tick_params(axis='x', labelsize=10)
axes[0,1].set_ylabel('Days', labelpad=10,size=15)
axes[0,1].tick_params(axis='y', labelsize=15)

axes[1,1].set_xlabel('Date', labelpad=5,size=15)
axes[1,1].tick_params(axis='x', labelsize=10)
axes[1,1].set_ylabel('human t_pred - rational prior (Days)', labelpad=10,size=15)
axes[1,1].tick_params(axis='y', labelsize=15)

axes[1,0].set_xlabel('human horizon (Days)', labelpad=5,size=15)
axes[1,0].tick_params(axis='x', labelsize=15)
axes[1,0].set_ylabel('human t_pred - rational prior (Days)', labelpad=10,size=15)
axes[1,0].tick_params(axis='y', labelsize=15)

axes[0,0].set_xlabel('human t_pred (Days)', labelpad=5,size=15)
axes[0,0].tick_params(axis='x', labelsize=15)
axes[0,0].set_ylabel('rational prior (Days)', labelpad=10,size=15)
axes[0,0].tick_params(axis='y', labelsize=15)

plt.subplots_adjust(wspace=.2,hspace=0.3)

#plt.savefig(f'Study1_ExplanatoryScatter_All_S.png', dpi=300, transparent=False, bbox_inches='tight')
'''END ALL S PLOTTING'''



'''PANELS OF INDIVIDUALS'''
'''ALL GPs on ONE PLOT'''
fig_ax_dim_x=2
fig_ax_dim_y=3
fig, axes = plt.subplots(fig_ax_dim_y,fig_ax_dim_x,figsize=(8,8),sharex=True,sharey=True)
leg_x = .95
leg_y = 1
#PANEL 0,0
gp_ordering = np.array([8,7,6,5,4,3])
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
        #axes[k,j].plot(grouped.p_dur.mean().rolling(4).mean(),color='black',label='human horizon',dashes=(0,0,2,2))
        axes[k,j].scatter(gt_S.index,gt_S,color='black',label='ground truth horizon',marker='p',s=10,alpha=0.45)
        if ((k==0) & (j==0)):
            axes[k,j].legend(bbox_to_anchor=(leg_x,leg_y), loc=1, frameon=False, fontsize=7)
        axes[k,j].set_ylim(0, 86)
        #axes[k,j].axvline(x=datetime.strptime('2021-12-03','%Y-%m-%d'),c='black',dashes=(6,6,6,6),linewidth=1)
        #axes[k,j].axvline(x=datetime.strptime('2021-12-24','%Y-%m-%d'),c='black',dashes=(6,6,6,6),linewidth=1)
        #axes[k,j].axvline(x=datetime.strptime('2022-01-14','%Y-%m-%d'),c='black',dashes=(6,6,6,6),linewidth=1)
        axes[k,j].text(datetime.strptime('2021-12-01','%Y-%m-%d'),77,'P'+str(gp_counter+1),fontsize=10)
        gp_counter += 1
#LABELS
axes[2,0].set_xlabel('Date', labelpad=10,size=10)
axes[2,0].tick_params(axis='x', labelsize=5)
axes[2,1].set_xlabel('Date', labelpad=10,size=10)
axes[2,1].tick_params(axis='x', labelsize=5)

axes[0,0].set_ylabel('Days', labelpad=10,size=10)
axes[1,0].set_ylabel('Days', labelpad=10,size=10)
axes[2,0].set_ylabel('Days', labelpad=10,size=10)
plt.subplots_adjust(wspace=.1,hspace=0.1)
#plt.savefig(f'Single_Subjects.png', dpi=300, transparent=False, bbox_inches='tight')


'''SUMMARY OF PANELS OF INDIVIDUALS'''
marker_list = ['x','+','o','v','^','*']

'''Date Summary'''
for k in range(0,len(gp_ordering)):
    gp_counter = k
    
    gp_no = group_list[gp_ordering[gp_counter]]
    i = catch_groups[gp_ordering[gp_counter]]
    i['hum_minus_prior'] = i.hum - i.prior
    '''aVE OVER DAY'''
    grouped = i.groupby(level=0)
    plt.scatter(i.index,i.hum_minus_prior,color='black',label='P' + str(k+1),marker=marker_list[k],s=50,alpha=0.15)
    #axes[k,j].plot(grouped.hum_minus_prior.mean().rolling(4).mean(),color='black')
    if (gp_counter+1 == 1):
        plt.plot(i.hum_minus_prior,color='black',dashes=(1,1,1,1),alpha=.7,linewidth=0.8)
    if (gp_counter+1 == 5):
        plt.plot(i.hum_minus_prior,color='red',dashes=(2,2,2,2),alpha=.7,linewidth=0.8)
    if (gp_counter+1 == 6):
        plt.plot(i.hum_minus_prior,color='green',dashes=(3,3,3,3),alpha=.7,linewidth=0.8)

plt.axvline(x=datetime.strptime('2022-01-09','%Y-%m-%d'),c='black',linewidth=1.6,alpha=0.3)
plt.legend(frameon=False, fontsize=11)
plt.xlabel('Date', labelpad=5,size=15)
plt.tick_params(axis='x', labelsize=8)
plt.ylabel('human t_pred - rational prior (Days)', labelpad=10,size=15)
plt.tick_params(axis='y', labelsize=15)
plt.savefig('Single_Ss_Date_Summary.png',dpi=300,transparent=False, bbox_inches='tight')


'''Horizon Summary'''
for k in range(0,len(gp_ordering)):
    gp_counter = k
    
    gp_no = group_list[gp_ordering[gp_counter]]
    i = catch_groups[gp_ordering[gp_counter]]
    i['hum_minus_prior'] = i.hum - i.prior
    '''aVE OVER DAY'''
    grouped = i.groupby(level=0)
    plt.scatter(i.p_dur,i.hum_minus_prior,color='black',label='P' + str(k+1),marker=marker_list[k],s=50,alpha=0.25)
    
plt.axvline(x=5,c='black',dashes=(2,2,2,2),linewidth=2,alpha=0.5)
plt.legend(frameon=False, fontsize=11)
plt.xlabel('human horizon (Days)', labelpad=5,size=15)
plt.tick_params(axis='x', labelsize=15)
plt.ylabel('human t_pred - rational prior (Days)', labelpad=10,size=15)
plt.tick_params(axis='y', labelsize=15)
#plt.savefig('Single_Ss_Horiz_Summary.png',dpi=300,transparent=False, bbox_inches='tight')












'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
EXTRA FOR EXPLORATORY AND SANITY CHECKS
'''

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
        axes[k,j].scatter(i.index,i.hum_minus_prior,color='black',label='t_predicted - prior',marker='^',s=50,alpha=0.25)
        axes[k,j].plot(grouped.hum_minus_prior.mean().rolling(4).mean(),color='black')
        #axes[k,j].text(60,10,'P'+str(gp_counter+1),fontsize=10)
        #axes[k,j].axvline(x=5,c='black',dashes=(2,2,2,2),linewidth=2,alpha=0.5)
        gp_counter += 1
plt.subplots_adjust(wspace=.1,hspace=0.1)
plt.savefig(f'Tmp2.png', dpi=300, transparent=False, bbox_inches='tight')
'''END QUICK CHECK'''




#EOF