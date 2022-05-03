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

data_in = '/Users/biocomplexity/Projects/SocioCognitiveModeling/Metaculus_CogModeling/simulations/Preprocessing'
priors_in = '/Users/biocomplexity/Projects/SocioCognitiveModeling/Metaculus_CogModeling/simulations/Priors'
outfile_name = 'sim_out_highest'

'''LOAD PRIORS'''
posteriors = pd.read_pickle(f'{priors_in}/catch_all_prior_over_all_t_out')
prior_means = pd.read_pickle(f'{priors_in}/catch_prior_index_out')
prior_dist = pd.read_pickle(f'{priors_in}/catch_all_dist_prior_over_all_mean_out')

'''GENERATE_DENSITY'''#FROM FILE.1 in sandbox
dist_prior = prior_dist[30]
dist_mean = prior_means[30]
#COMPUTE PROBS FOR EACH EVENT
dist_prior_event_probs = pd.Series(dist_prior).value_counts()/pd.Series(dist_prior).value_counts().sum()
dist_prior_event_probs = dist_prior_event_probs.sort_index()
plt.subplot(2,3,1)
plt.hist(dist_prior_event_probs,bins=30)
plt.subplot(2,3,2)
plt.plot(dist_prior_event_probs)
plt.subplot(2,3,3)
plt.hist(dist_prior,bins=30)
#SUMS TO ONE
dist_prior_event_probs.sum()


'''GENERATE POSTERIORS'''
catch_posteriors = []
catch_t = []
for i in range(30,51,10):
    x = bayes.compute_posterior(i,dist_prior_event_probs,1000)
    x = pd.Series(x,index=dist_prior_event_probs.index)
    catch_posteriors.append(x)
    catch_t.append(i)

'''CAN ADD THIS STUFF LATER'''
prior_med = bayes.median_of_dist(dist_prior_event_probs)
decision_med = bayes.median_of_dist(x)
plt.title('Decision Process with 40 Days as Input')
plt.plot(dist_prior_event_probs,label=f'prior, median={prior_med}')
plt.plot(x,label=f'decision {i} -> median={decision_med}')
plt.axvline(x=prior_med, c='b',dashes=(2,2,2,2),linewidth=1)
plt.axvline(x=decision_med, c='b',linewidth=1)
plt.legend()

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

ax.plot(dist_prior_event_probs,color='black',label='prior (E[x]=50)',linewidth=2)
ax.plot(catch_posteriors[0],color='black',label='t=30',dashes=(0,2,2,2))
ax.plot(catch_posteriors[1],color='black',label='t=40',dashes=(0,0,2,2))
ax.plot(catch_posteriors[2],color='black',label='t=50',dashes=(0,0,1,1))
ax.legend(bbox_to_anchor=(.85, .85), loc=1, frameon=False, fontsize=20)
plt.savefig('TheoryPosteriorDists_1', dpi=300, transparent=False, bbox_inches='tight')
plt.show()

#EOF