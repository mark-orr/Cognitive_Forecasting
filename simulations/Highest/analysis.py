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
ax.plot(df_test_mean_forward.delta,color='black',label='human t_total delta',dashes=(6,6,6,6),alpha=0.3)

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

fwd_value = 3
test_mean_forward = test_mean.copy()
test_mean_forward = test_mean_forward.reset_index(drop=True)
test_mean_forward = test_mean_forward[:-fwd_value].reset_index(drop=True)
test_mean_forward = pd.concat([pd.Series(np.full(fwd_value,fill_value=np.NaN)),test_mean_forward],axis=0)
test_mean_forward.index = test_mean.index
test_mean_forward.name = 'forward'

df_test_mean_forward = pd.concat([test_mean,test_mean_forward],axis=1)
df_test_mean_forward['delta'] = df_test_mean_forward.hum - df_test_mean_forward.forward
plt.plot(df_test_mean_forward.delta)


'''ADD EPI'''
'''DATA FROM VIRGINIA'''
#SRINI RECOMMEND:
'''/project/biocomplexity/COVID-19_commons/data/VDH_public/VDH-COVID-19-PublicUseDataset-Cases.csv'''



#THIS IS COVID CAST EXAMPLE I THINK (DATA IN SANDBOX)
df_va_ts = pd.read_csv('VA_Case_timeSeries.csv')
#MAKE DELTA FOR index > 0
S_cases = df_va_ts.total_cases
S_delta = np.array([])
S_delta = np.append(S_delta,10)
for i in range(1,len(S_cases)):
    delta_cases = S_cases[i]-S_cases[i-1]
    S_delta = np.append(S_delta,delta_cases)
S_cases.iloc[-1] - S_delta.sum()#should equal 57
S_cases.index#JUST CHECKING
df_va_ts.index#JUST CHECKING
df_va_ts['incident_cases'] = S_delta
df_va_ts.report_date[485:640]#HOMING IN
df_ts_delta = df_va_ts['incident_cases'][485:640]
df_ts_delta.index = df_va_ts.report_date[485:640]
df_ts_delta.plot()









'''
NOTE TO DO
MAKE PANELS AND PLOT EACH ONE OF THESE
'''
gp_no = group_list[2]
i = catch_groups[2]
ax.scatter(i.index, i.p_dur,linewidth=2,color=colors[1],label=f'{gp_no}')
ax.plot(i.prior.rolling(2).mean(),color='black')
ax.plot(i.hum.rolling(2).mean(),color='red')
ax.plot(i.p_dur.rolling(2).mean(),color=colors[1])

gp_no = group_list[3]
i = catch_groups[3]
ax.scatter(i.index, i.p_dur,linewidth=2,color=colors[2],label=f'{gp_no}')
ax.plot(i.prior.rolling(2).mean(),color=colors[2])
ax.plot(i.hum.rolling(2).mean(),color='red')
ax.plot(i.p_dur.rolling(2).mean(),color=colors[2])

gp_no = group_list[4]
i = catch_groups[4]
ax.scatter(i.index, i.p_dur,linewidth=2,color=colors[3],label=f'{gp_no}')
ax.plot(i.prior.rolling(2).mean(),color=colors[3])
ax.plot(i.hum.rolling(2).mean(),color='red')
ax.plot(i.p_dur.rolling(2).mean(),color=colors[3])

gp_no = group_list[5]
i = catch_groups[5]
ax.scatter(i.index, i.p_dur,linewidth=2,color=colors[4],label=f'{gp_no}')
ax.plot(i.prior.rolling(2).mean(),color='black')
ax.plot(i.hum.rolling(2).mean(),color='red')
ax.plot(i.p_dur.rolling(2).mean(),color=colors[4])

gp_no = group_list[6]
i = catch_groups[6]
ax.scatter(i.index, i.p_dur,linewidth=2,color=colors[5],label=f'{gp_no}')
ax.plot(i.prior.rolling(2).mean(),color='black')
ax.plot(i.hum.rolling(2).mean(),color='red')
ax.plot(i.p_dur.rolling(2).mean(),color=colors[5])

gp_no = group_list[7]
i = catch_groups[7]
ax.scatter(i.index, i.p_dur,linewidth=2,color=colors[6],label=f'{gp_no}')
ax.plot(i.prior.rolling(2).mean(),color='black')
ax.plot(i.hum.rolling(2).mean(),color='red')
ax.plot(i.p_dur.rolling(2).mean(),color=colors[6])

gp_no = group_list[8]
i = catch_groups[8]
ax.scatter(i.index, i.p_dur,linewidth=2,color=colors[7],label=f'{gp_no}')
ax.plot(i.prior.rolling(2).mean(),color='black')
ax.plot(i.hum.rolling(2).mean(),color='red')
ax.plot(i.p_dur.rolling(2).mean(),color=colors[7])


ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
plt.savefig('Final_Plot.png', dpi=300, transparent=False, bbox_inches='tight')
plt.show()
















#EOF