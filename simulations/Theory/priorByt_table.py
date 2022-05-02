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

for i in range(0,len(posteriors)):  
    print(prior_means[i],': ',posteriors[i][:prior_means[i]])
    plt.plot(posteriors[i][:prior_means[i]])
    
#PUT INTO NP ARRAY FOR HEAT MAP
catch_array = np.zeros((len(posteriors),len(posteriors)+19))

for i in range(0,len(posteriors)):
    sub = posteriors[i][:prior_means[i]]
    catch_array[i][0:len(sub)] = sub
    catch_array[i][len(sub):] = None
    
plt.imshow(catch_array,cmap='RdYlBu')
plt.xlabel('User Index')
plt.ylabel('Question Index')
plt.title('All Users No. Responses Over All Questions')

'''NEED SUBTRACTIVE FROM MEDIAN OF PRIOR'''
'''IS PRIOR MEANS'''
catch_array_2 = catch_array.copy()
for i in range(0, len(prior_means)):
    catch_array_2[i][:] = catch_array_2[i]-prior_means[i]
    

plt.imshow(catch_array_2,cmap='RdYlBu')
plt.xlabel('t_i')
plt.ylabel('posterior | t_total minus median of prior')
plt.colorbar()
#plt.savefig('All_Users_Respone_Heatmap.png',dpi=250)



'''NEXT, MAKE IT GRAY SCALE AND SAME FORMAT AS OTHERS'''
'''ALL GPs on ONE PLOT'''
'''FANCIER METHOD'''
mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
#font_names = [f.name for f in fm.fontManager.ttflist]
#print(font_names)

plt.style.use('default')

fig = plt.figure(figsize=(7, 5))

ax = fig.add_axes([0, 0, 2, 2])
#x.set_title(f'???',size=20)
ax.set_xlabel('t_i', labelpad=10,size=20)
ax.set_ylabel('Median of Prior', labelpad=10,size=20)

#ax.set_ylim(-10, 90)
ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.set_yticks([5,30,55,80,105,130,155])
ax.set_yticklabels(['25','50','75','100','125','150','175'])


ax.imshow(catch_array_2,cmap='RdYlBu')
fig.colorbar(ax.imshow(catch_array_2,cmap='RdYlBu'))
for i in range(25,181,25):
    ax.axvline(x=i,c='black',dashes=(3,3,3,3),linewidth=1)
    ax.axhline(y=i-20,c='black',dashes=(3,3,3,3),linewidth=1)
plt.savefig('TheoryHeatMap_1', dpi=300, transparent=False, bbox_inches='tight')
plt.show()
'''THIS IS THE HEAT MEASURE (Posterior | prior & t_i) - median of prior)'''