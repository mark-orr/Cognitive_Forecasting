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
    

group_list
'''INITIAL PLOTS FOR WRITING'''
gp_no = group_list[0]
i = catch_groups[0]  
plt.scatter(i.index,i.p_dur,label='raw')
i.prior.rolling(4).mean().plot(label='optimal')
i.hum.rolling(4).mean().plot(label='human')
i.p_dur.rolling(4).mean().plot(label='raw')
plt.title(f'Human Judgments and Optimal (GP={gp_no}')
plt.ylabel('Days')
plt.xlabel('DateTime')
plt.axvline(x=datetime.strptime('2021-12-24','%Y-%m-%d'),c='b',dashes=(2,2,2,2),linewidth=1)
plt.axvline(x=datetime.strptime('2022-01-14','%Y-%m-%d'),c='b',dashes=(2,2,2,2),linewidth=1)
plt.legend()
    

    
    
    
'''FANCIER METHOD'''
mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2

import matplotlib.font_manager as fm   
from pylab import cm
import matplotlib as mpl

font_names = [f.name for f in fm.fontManager.ttflist]
print(font_names)
# Generate 2 colors from the 'tab10' colormap
#colors = cm.get_cmap('tab10', 2)
colors = ['#ab0049', '#ab7972', '#abc07e', '#d1ff00']
# Create figure and add axes object
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
# Plot and show our data
ax.scatter(i.index, i.p_dur)
ax.plot(i.prior.rolling(4).mean())
plt.show()
    
gp_no = group_list[0]
i = catch_groups[0]

fig = plt.figure(figsize=(7, 5))

ax = fig.add_axes([0, 0, 2, 2])
ax.set_title(f'Human Judgments and Optimal (GP={gp_no})')
ax.set_xlabel('Date', labelpad=10)
ax.set_ylabel('Days', labelpad=10)

ax.set_ylim(0, 90)
ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


gp_no = group_list[0]
i = catch_groups[0]
ax.scatter(i.index, i.p_dur,linewidth=2,color=colors[0],label='raw')
ax.plot(i.prior.rolling(21).mean(),label='optimal',color=colors[1])
ax.plot(i.hum.rolling(21).mean(),label='human',color=colors[2])
ax.plot(i.p_dur.rolling(21).mean(),label='raw',color=colors[0])

gp_no = group_list[6]
i = catch_groups[6]
ax.scatter(i.index, i.p_dur,linewidth=2,color=colors[3],label='raw')
ax.plot(i.prior.rolling(2).mean(),label='optimal',color=colors[1])
ax.plot(i.hum.rolling(2).mean(),label='human',color=colors[2])
ax.plot(i.p_dur.rolling(2).mean(),label='raw',color=colors[3])

ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
plt.show()


plt.savefig('Final_Plot.png', dpi=300, transparent=False, bbox_inches='tight')



'''ALL GPs on ONE PLOT'''
#COLORBLIND SAFE
colors = ['#ff0000', '#a68465', '#bb8a86', '#da8ba2', '#92a6b1', '#778688', '#5c6673', '#17439b']
#colors = ['#ee5553', '#ca6357', '#a9685d', '#8b6865', '#70656e', '#576079', '#405886', '#284e95', '#0041a6']
fig = plt.figure(figsize=(7, 5))

ax = fig.add_axes([0, 0, 2, 2])
ax.set_title(f'Human Judgments and Optimal All Individuals')
ax.set_xlabel('Date', labelpad=10)
ax.set_ylabel('Days', labelpad=10)

ax.set_ylim(0, 90)
ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

#gp_no = group_list[0]
#i = catch_groups[0]
#ax.scatter(i.index, i.p_dur,linewidth=2,color=colors[0],label='raw')
#ax.plot(i.prior.rolling(21).mean(),label='optimal',color='black')
#ax.plot(i.hum.rolling(21).mean(),label='human',color='red')
#ax.plot(i.p_dur.rolling(21).mean(),label='raw',color=colors[0])

gp_no = group_list[1]
i = catch_groups[1]
ax.scatter(i.index, i.p_dur,linewidth=2,color=colors[0],label=f'{gp_no}')
ax.plot(i.prior.rolling(2).mean(),color='black')
ax.plot(i.hum.rolling(2).mean(),color='red')
ax.plot(i.p_dur.rolling(2).mean(),color=colors[0])

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