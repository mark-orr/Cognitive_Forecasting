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


'''
NOTES:
1. FOR MANUSCRIPT, ONE OF THE LAST CODE BLOCKS (LABEL: PRIOR DROP COMPARISON) HERE COMPUTES THE THEORETICAL 
CONSTANT DECISION WITH THE PARAMETERS OF THE DATA.  IT WILL OUTPUT A CSV FOR INGESTION
BY THE ANALYSIS.PY FILE IN /SIMULATIONS/HIGHEST.  MAY NOT USE IT IN FINAL V OF MANU, 
BUT ITS HERE TO TRY.
2. Another LATER CODE BLOCK (LABEL: ANNEALING) FINDS OPTIMAL SCHEDULE OF
T_PREDICTED SO THAT DIFF BETWEEN PRIOR LAMBDA
'''


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

'''MAKE FANCY'''
mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
#font_names = [f.name for f in fm.fontManager.ttflist]
#print(font_names)
'''ALL GPs on ONE PLOT'''
plt.style.use('default')

fig = plt.figure(figsize=(7, 5))

ax = fig.add_axes([0, 0, 2, 2])
ax.set_xlabel('Date', labelpad=10,size=20)
ax.set_ylabel('Days', labelpad=10,size=20)

ax.set_ylim(0, 55)
ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


ax.scatter(S_all.index, S_all.p_dur,color='black',label='human horizon',marker='+',s=150,alpha=0.25)
ax.plot(S_all.prior,color='black',label='prior',dashes=(0,2,2,2))
ax.plot(S_all.hum,color='black',label='human t_total')
ax.plot(S_all.p_dur,color='black',label='human horizon',dashes=(0,0,2,2))

ax.axvline(x=datetime.strptime('2022-02-16','%Y-%m-%d'),c='black',dashes=(8,8,8,8),linewidth=1)
#ax.axhline(y=44,c='black',dashes=(8,8,8,8),linewidth=1)
ax.axvline(x=datetime.strptime('2022-02-18','%Y-%m-%d'),c='black',dashes=(8,8,8,8),linewidth=1)
#ax.axhline(y=32,c='black',dashes=(8,8,8,8),linewidth=1)

ax.legend(bbox_to_anchor=(.8, .8), loc=1, frameon=False, fontsize=20)
plt.savefig(f'Theory_PriorShift_2.png', dpi=300, transparent=False, bbox_inches='tight')
plt.show()



'''UNDER CONSTRUCITON'''
'''MAKE PANELS FOR ABOVE + CONSTANT PRIOR'''
new_data = pd.Series(np.array([50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 51., 51., 51., 52., 52., 53., 53., 54., 54.]),dtype=int)

df_S_w2['prediction_int'] = new_data
df_S_w2['prediction_horiz_int'] = df_S_w2.prediction_int - df_S_w2.t_int

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
#plot_this.name='prior'
#replacing plot_this with constant
constant_prior = np.repeat(50,len(index_for_all))
S_constant_prior = pd.Series(constant_prior,index=df_S_w2.decision_date.reset_index(drop=True),name='prior')
S_er.name='err'

S_all_2 = pd.concat([S_ts, S_hp, S_pd, S_constant_prior, S_er], axis=1)
#S_all.to_pickle(f'S_all_{outfile_group}.csv')

S_all_2.plot()


'''NOTE, NOW HAVE BEST PRIORS ESTIMATED FOR BOTH, 
S_all (constant t_total) and S_all_2 (const prior)'''
'''NOTE, we can see when t_total = t_i, that is when the gamme is over
For our priors (numerical dists), at mean of 50, this happens at t=70 with t_total=70,
could also put it in terms of horizon, which starts to slow its rate....
This gives a differennt feel to the argument...because on does know t_i and their estimate, 
as they get close, either prior drops or horison starts to increase.  What do we see in these data?

THIS CODE IS WHAT WE USED IN ./priorsByt_table.py to get t=70.
for i in range(0,75): print('t: ',i+1,'post: ',posteriors[30][i])
'''
#SWAP IN DAYS ON INDEX
S_all.index = np.arange(50)+1
S_all_2.index = np.arange(50)+1


fix, axes = plt.subplots(2,1,figsize=(5,7),sharex=True,sharey=True)
leg_x = 1
leg_y = 1
gp_list = ['Constant Decision','Constant Prior']

axes[0].scatter(S_all.index, S_all.p_dur,color='black',label='human horizon',marker='+',s=150,alpha=0.25)
axes[0].plot(S_all.prior,color='black',label='rational prior',dashes=(0,2,2,2))
axes[0].plot(S_all.hum,color='black',label='human t_predicted')
axes[0].plot(S_all.p_dur,color='black',label='human horizon',dashes=(0,0,2,2))
axes[0].set_ylim(0,58)
#axes[0].text(datetime.strptime('2022-01-01','%Y-%m-%d'),53,gp_list[0],fontsize=10)
axes[0].legend(bbox_to_anchor=(.4, .4), loc=1, frameon=False, fontsize=8)
axes[0].tick_params(axis='x', labelsize=6)
axes[0].tick_params(axis='x', labelsize=10)
axes[0].tick_params(axis='y', labelsize=10)

axes[1].scatter(S_all_2.index, S_all_2.p_dur,color='black',label='human horizon',marker='+',s=150,alpha=0.25)
axes[1].plot(S_all_2.prior,color='black',label='rational prior',dashes=(0,2,2,2))
axes[1].plot(S_all_2.hum,color='black',label='human t_predicted')
axes[1].plot(S_all_2.p_dur,color='black',label='human horizon',dashes=(0,0,2,2))
axes[1].set_ylim(0,58)
#axes[1].text(datetime.strptime('2022-01-01','%Y-%m-%d'),53,gp_list[1],fontsize=10)
axes[1].legend(bbox_to_anchor=(.4, .4), loc=1, frameon=False, fontsize=8)
axes[1].tick_params(axis='x', labelsize=10)
axes[1].tick_params(axis='y', labelsize=10)

#LABELS
axes[1].set_xlabel('t_past (Minutes)', labelpad=10,size=10)
axes[0].set_ylabel('Minutes', labelpad=10,size=10)
axes[1].set_ylabel('Minutes', labelpad=10,size=10)

plt.subplots_adjust(wspace=.1,hspace=0.1)
plt.savefig(f'Theory_PriorShift_1_2panel.png', dpi=300, transparent=False, bbox_inches='tight')











'''NOW PLOT PRIOR DISTRIBUTION SHIFT'''

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
t=49, prior=32, decision=50, posterior=50
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



'''GOOD PLOT BUT FOR GETTTING DATA STRUCTURE CORRECT'''
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


'''MAKE PANELS'''
fix, axes = plt.subplots(3,1,figsize=(5,7),sharex=True,sharey=False)
leg_x = 1
leg_y = 1
gp_list = ['t_past=41,prior=50','t_past=47,prior=44','t_past=49,prior=32']

for i in range(0,3):
    axes[i].plot(catch_dist_priors[i],color='black',label='prior',linewidth=2)
    axes[i].plot(catch_posteriors[i],color='black',label='posterior',dashes=(0,2,2,2))
    axes[i].axvline(x=bayes.median_of_dist(catch_posteriors[i]),c='black',dashes=(1,1,1,1),linewidth=1.5,alpha=0.3)
    axes[i].set_ylim(0,0.15)
    axes[i].text(15,.13,gp_list[i],fontsize=10)
    axes[i].legend(bbox_to_anchor=(leg_x,leg_y), loc=1, frameon=False, fontsize=10)

#LABELS
axes[2].set_xlabel('t_total', labelpad=10,size=10)
axes[0].set_ylabel('Probability', labelpad=10,size=10)
axes[1].set_ylabel('Probability', labelpad=10,size=10)
axes[2].set_ylabel('Probability', labelpad=10,size=10)

plt.subplots_adjust(wspace=.1,hspace=0.1)
plt.savefig(f'Theory_PriorShift_1.png', dpi=300, transparent=False, bbox_inches='tight')











'''
LABEL: PRIOR DROP COMPARISON
LAST CODE BLOCK, SEE NOTE IN HEADER OF THIS FILE
'''
'''MAKE FAKE DATA'''
judgment_offset = 0
index_for_all = pd.date_range('2021-11-30','2022-01-14')
S_0 = pd.Series(pd.date_range('2021-11-30','2022-01-14'),name='decision_date')
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

ForExportToAnalysis = pd.Series(S_all.prior)
ForExportToAnalysis.name = 'const_dec_prior'
ForExportToAnalysis.to_csv('ConstantDecisionPrior_Comp.csv',index=True)









'''
LABEL: ANNEALING

'''
'''UNDER CONSTRUCITON'''

'''MAKE FAKE DATA'''
judgment_offset = 0
index_for_all = pd.date_range('2022-01-01','2022-03-15')
S_0 = pd.Series(pd.date_range('2022-01-01','2022-03-15'),name='decision_date')
S_1 = pd.Series(range(1,1+len(index_for_all)),name='t_int')
S_2 = pd.Series(np.repeat(len(index_for_all)-judgment_offset,len(index_for_all)),name='prediction_int')
S_3 = pd.Series((len(index_for_all)-judgment_offset)-S_1,name='prediction_horiz_int')

df_use_all = pd.concat([S_0,S_1,S_2,S_3],axis=1)
len(index_for_all)



'''COMPUTE BEST PRIORS'''
'''ALL Ss TOGETHER'''
outfile_group = 'fake_data'
df_S_w2 = df_use_all.copy()


#ALSO INTERESTING
new_data = pd.Series(np.array([50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 
50., 50., 50., 51., 51., 52., 52., 53., 53., 
54., 54., 55., 55., 56., 56., 57., 57., 58., 58.,
59., 59., 60., 60., 61., 61., 62., 62., 63., 63.,
64., 64., 65., 65.,]),dtype=int)

new_data = pd.Series(np.array([50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 
                               50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 
                               50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 
                               50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 
                               50., 50., 
                               51., 52., 53., 53., 
                               54., 55., 56., 56., 
                               57., 58., 59., 59., 
                               60., 61., 62., 62., 
                               63., 64., 65., 65., 
                               66., 67., 68., 68., 
                               69., 70., 71., 71., 
                               72., 73., 74., 74.,]),dtype=int)

#new_data = pd.Series(np.repeat(50,74))
'''WORKING ON FUNCTIONS'''
np.array(np.exp(tmp_arr)*(32/(np.exp(32))),dtype=int)
new_data = pd.Series(np.array([50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 
                               50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 
                               50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 
                               50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 
                               50., 50., 
                               51., 52., 53., 54., 
                               55., 56., 57., 58., 
                               59., 60., 61., 62., 
                               63., 64., 65., 66., 
                               67., 68., 69., 70., 
                               71., 72., 73., 74., 
                               75., 76., 77., 78., 
                               79., 80., 81., 82.,]),dtype=int)

new_data = pd.Series(np.array([50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 
                               50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 
                               50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 
                               50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 
                               50., 50., 
                               51., 52., 53., 54., 
                               55., 56., 57., 57., 
                               58., 59., 60., 61., 
                               62., 63., 65., 65., 
                               66., 67., 68., 69., 
                               70., 71., 75., 75., 
                               76., 77., 78., 79., 
                               80., 86., 86., 86.,]),dtype=int)


df_S_w2['prediction_int'] = new_data
df_S_w2['prediction_horiz_int'] = df_S_w2.prediction_int - df_S_w2.t_int

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

for j in posteriors[20:60]: #LOOP OVER PRIORS
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
df_error.columns = prior_means[20:60]
plot_this = df_error.abs().idxmin(axis=1)

#USEFUL VIS
S_ts = pd.Series(catch_all_t_over_t,index=df_S_w2.decision_date.reset_index(drop=True),name='t')
S_hp = pd.Series(catch_all_human_pred_over_t,index=df_S_w2.decision_date.reset_index(drop=True),name='hum')
S_pd = pd.Series(catch_all_p_dur_over_t,index=df_S_w2.decision_date.reset_index(drop=True),name='p_dur')
S_er = df_error.abs().min(axis=1)
plot_this.name='prior'
#replacing plot_this with constant
#constant_prior = np.repeat(50,len(index_for_all))
#S_constant_prior = pd.Series(constant_prior,index=df_S_w2.decision_date.reset_index(drop=True),name='prior')
S_er.name='err'

#S_all_2 = pd.concat([S_ts, S_hp, S_pd, S_constant_prior, S_er], axis=1)
S_all_2 = pd.concat([S_ts, S_hp, S_pd, plot_this, S_er], axis=1)
#S_all.to_pickle(f'S_all_{outfile_group}.csv')

S_all_2.plot()

plt.plot(S_all_2.prior.rolling(4))



#EOF
