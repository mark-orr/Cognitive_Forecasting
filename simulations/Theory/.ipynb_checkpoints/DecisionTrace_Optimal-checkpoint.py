import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import pickle

from pylab import cm
import matplotlib as mpl

from imp import reload
import sys
sys.path.insert(1,'/Users/biocomplexity/Projects/SocioCognitiveModeling/Metaculus_CogModeling/bayes_pack_testing')
import bayes
import b_fit

reload(bayes)

priors_in = '/Users/biocomplexity/Projects/SocioCognitiveModeling/Metaculus_CogModeling/simulations/Priors'

'''LOAD PRIORS'''
posteriors = pd.read_pickle(f'{priors_in}/catch_all_prior_over_all_t_out')
prior_means = pd.read_pickle(f'{priors_in}/catch_prior_index_out')

'''MAKE FAKE DATA'''
judgment_offset = 0
index_for_all = pd.date_range('2022-01-01','2022-02-19')
S_0 = pd.Series(pd.date_range('2022-01-01','2022-02-19'),name='decision_date')
S_1 = pd.Series(range(1,1+len(index_for_all)),name='t_int')
S_2 = pd.Series(np.repeat(len(index_for_all)-judgment_offset,len(index_for_all)),name='prediction_int')
S_3 = pd.Series((len(index_for_all)-judgment_offset)-S_1,name='prediction_horiz_int')

df_use_all = pd.concat([S_0,S_1,S_2,S_3],axis=1)
len(index_for_all)



'''COMPUTE BEST PRIORS'''
'''ALL Ss TOGETHER'''
outfile_group = 'fake_data'
df_S_w2 = df_use_all.copy()
'''
IV.  GENERATE BEST PRIORS FOR W2
'''    
catch_all_t_over_t_over_p = []
catch_all_optimal_pred_over_t_over_p = []
catch_tmp_optimal_over_p = []
catch_all_human_pred_over_t_over_p = []
catch_all_error_over_t_over_p = []
catch_all_date_over_t_over_p = []
catch_all_p_dur_over_t_over_p = []

for j in posteriors: #LOOP OVER PRIORS
    print('J is:',j)
    print('NEW PRIOR')
    print('NEW PRIOR')
    catch_all_t_over_t = ([])
    catch_all_optimal_pred_over_t = ([])
    catch_tmp_optimal_over_t = ([])
    catch_all_human_pred_over_t = ([])
    catch_all_error_over_t = ([])
    catch_all_date_over_t = ([])
    catch_all_p_dur_over_t = ([])

    for i in range(0,len(df_S_w2)):#CAPTURE HUMAN DATA T AND PRED
        t = df_S_w2.t_int.iloc[i] #P
        
        p_dur = df_S_w2.prediction_horiz_int.iloc[i]
        catch_all_p_dur_over_t = np.append(catch_all_p_dur_over_t,p_dur)
        
        print('t',t)
        catch_all_t_over_t = np.append(catch_all_t_over_t,t)
        #optimal_pred = catch_all_prior_over_all_t[0][t-1]#index zero is t=1
        print('LEN of J',len(j))
        if (t>0) & (t<=len(j)):#t-time is j[j.index+1]
            print('T IS WITHIN PRIOR,  GOOOD')
            optimal_pred = j[t-1]
        else:
            print('T IS GREATER THAN Ts in PRIOR')
            optimal_pred = j[-1]
        catch_all_optimal_pred_over_t = np.append(catch_all_optimal_pred_over_t,optimal_pred)
        print('optimal pred',optimal_pred)
        catch_tmp_optimal_over_t = np.append(catch_tmp_optimal_over_t,b_fit.find_optimal(t,j))
        
        human_pred = df_S_w2.prediction_int.iloc[i]
        catch_all_human_pred_over_t = np.append(catch_all_human_pred_over_t,human_pred)
        print('human pred', human_pred)
        error = human_pred - optimal_pred
        print('error',error)
        catch_all_error_over_t = np.append(catch_all_error_over_t,error)
        catch_all_date_over_t = np.append(catch_all_date_over_t,df_S_w2.decision_date.iloc[i])
        print('decision_date',df_S_w2.decision_date.iloc[i])
    
    catch_all_t_over_t_over_p.append(catch_all_t_over_t)
    catch_all_optimal_pred_over_t_over_p.append(catch_all_optimal_pred_over_t)
    catch_tmp_optimal_over_p.append(catch_tmp_optimal_over_t)
    catch_all_human_pred_over_t_over_p.append(catch_all_human_pred_over_t)
    catch_all_error_over_t_over_p.append(catch_all_error_over_t)
    catch_all_date_over_t_over_p.append(catch_all_date_over_t)
    catch_all_p_dur_over_t_over_p.append(catch_all_p_dur_over_t)
#TEST FOR FINDAL OPTIMAL FUNCTION SHOULD BE ZERO SUM
(np.array(catch_all_optimal_pred_over_t_over_p)-np.array(catch_tmp_optimal_over_p)).sum()

'''PICK BEST PRIOR'''
df_error = pd.DataFrame(catch_all_error_over_t_over_p).T
df_error.index = df_S_w2.decision_date.reset_index(drop=True)
df_error.columns = prior_means
plot_this = df_error.abs().idxmin(axis=1)

#USEFUL VIS
S_ts = pd.Series(catch_all_t_over_t,index=df_S_w2.decision_date.reset_index(drop=True),name='t')
S_hp = pd.Series(catch_all_human_pred_over_t,index=df_S_w2.decision_date.reset_index(drop=True),name='hum')
S_pd = pd.Series(catch_all_p_dur_over_t,index=df_S_w2.decision_date.reset_index(drop=True),name='p_dur')
S_er = df_error.abs().min(axis=1)
plot_this.name='prior'
S_er.name='err'

S_all = pd.concat([S_ts, S_hp, S_pd, plot_this, S_er], axis=1)
#S_all.to_pickle(f'S_all_{outfile_group}.csv')

S_all.plot()

'''NEXT STEPS:  
1. MAKE GRAPH NICE FROM ABOVE DATA, CAN USE HUMAN DATA GRAPH FORMAT
2. Make COMPANION GRAPH THAT SHOWS THE CHANGE IN PRIORS
AT THREE POINTS IN THESE SIMULATED DATA. STACKED VERTICALLY
THAT SHOWS HOW PRIOR --> POSTERIOR CHANGES FIXING THE JUDGMENT
AT 50 DAYS AS T PROGRESSES TO 50:  CAN USE THE Decision_process_illustrated.py

'''


'''LOAD PRIORS'''
prior_dist = pd.read_pickle(f'{priors_in}/catch_all_dist_prior_over_all_mean_out')
'''WILL CATCH STUFF FOR PLOTTING FOR EACH PRIOR'''
catch_posteriors = []
catch_t = []
catch_dist_priors = []


'''
t=41, prior=50, decision=50, posterior=50
'''
'''GENERATE_DENSITY'''#FROM FILE.1 in sandbox
dist_prior = prior_dist[30]
dist_mean = prior_means[30]
print(dist_mean)
#COMPUTE PROBS FOR EACH EVENT
dist_prior_event_probs = pd.Series(dist_prior).value_counts()/pd.Series(dist_prior).value_counts().sum()
dist_prior_event_probs = dist_prior_event_probs.sort_index()
#SUMS TO ONE
dist_prior_event_probs.sum()
catch_dist_priors.append(dist_prior_event_probs)
'''GENERATE POSTERIORS'''
t=41
x = bayes.compute_posterior(t,dist_prior_event_probs,1000)
x = pd.Series(x,index=dist_prior_event_probs.index)
catch_posteriors.append(x)
catch_t.append(t)


'''
t=47, prior=44, decision=50, posterior=50
'''
'''GENERATE_DENSITY'''#FROM FILE.1 in sandbox
dist_prior = prior_dist[24]
dist_mean = prior_means[24]
print(dist_mean)
#COMPUTE PROBS FOR EACH EVENT
dist_prior_event_probs = pd.Series(dist_prior).value_counts()/pd.Series(dist_prior).value_counts().sum()
dist_prior_event_probs = dist_prior_event_probs.sort_index()
#SUMS TO ONE
dist_prior_event_probs.sum()
catch_dist_priors.append(dist_prior_event_probs)
'''GENERATE POSTERIORS'''
t=47
x = bayes.compute_posterior(t,dist_prior_event_probs,1000)
x = pd.Series(x,index=dist_prior_event_probs.index)
catch_posteriors.append(x)
catch_t.append(t)


'''
t=47, prior=44, decision=50, posterior=50
'''
'''GENERATE_DENSITY'''#FROM FILE.1 in sandbox
dist_prior = prior_dist[12]
dist_mean = prior_means[12]
print(dist_mean)
#COMPUTE PROBS FOR EACH EVENT
dist_prior_event_probs = pd.Series(dist_prior).value_counts()/pd.Series(dist_prior).value_counts().sum()
dist_prior_event_probs = dist_prior_event_probs.sort_index()
#SUMS TO ONE
dist_prior_event_probs.sum()
catch_dist_priors.append(dist_prior_event_probs)
'''GENERATE POSTERIORS'''
t=49
x = bayes.compute_posterior(t,dist_prior_event_probs,1000)
x = pd.Series(x,index=dist_prior_event_probs.index)
catch_posteriors.append(x)
catch_t.append(t)



'''GOOD PLOT'''
mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
#font_names = [f.name for f in fm.fontManager.ttflist]
#print(font_names)

plt.style.use('default')

fig = plt.figure(figsize=(7, 5))

ax = fig.add_axes([0, 0, 2, 2])
#x.set_title(f'???',size=20)
ax.set_xlabel('t_total', labelpad=10,size=20)
ax.set_ylabel('Density', labelpad=10,size=20)

#ax.set_ylim(-10, 90)
ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.plot(catch_dist_priors[0],color='black',label='prior (E[x]=50)',linewidth=2)
ax.plot(catch_posteriors[0],color='black',label='t=41',dashes=(0,2,2,2))
ax.plot(catch_dist_priors[1],color='black',label='prior (E[x]=44)',linewidth=2)
ax.plot(catch_posteriors[1],color='black',label='t=47',dashes=(0,0,2,2))
ax.plot(catch_dist_priors[2],color='black',label='prior (E[x]=32)',linewidth=2)
ax.plot(catch_posteriors[2],color='black',label='t=49',dashes=(0,0,1,1))
ax.legend(bbox_to_anchor=(.85, .85), loc=1, frameon=False, fontsize=20)
ax.axvline(x=bayes.median_of_dist(catch_posteriors[0]),c='black',dashes=(0,2,2,2),linewidth=2,alpha=0.3)
ax.axvline(x=bayes.median_of_dist(catch_posteriors[1]),c='black',dashes=(0,0,2,2),linewidth=2,alpha=0.3)
ax.axvline(x=bayes.median_of_dist(catch_posteriors[2]),c='black',dashes=(0,0,1,1),linewidth=2,alpha=0.3)
#plt.savefig('TheoryPosteriorDists_2', dpi=300, transparent=False, bbox_inches='tight')
plt.show()


'''MAKE PANELS USING THIS TEMPLATE'''
fix, axes = plt.subplots(1,3,figsize=5,7,sharex=True,sharey=False)
leg_x = 1
leg_y = 1

'''TEMPLATE'''
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

