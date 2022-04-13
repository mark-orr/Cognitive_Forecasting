import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import pickle

import sys
sys.path.insert(1,'/Users/biocomplexity/Projects/SocioCognitiveModeling/Metaculus_CogModeling/bayes_pack')
import bayes
'''
PURPOSE OF FILE IS TO GENERATE A SHARED SET OF PRIORS
FOR ALL SIMULATIONS
'''

#FOR PRIOR
N = 1000

'''COMPUTE PRIORS'''
catch_all_prior_over_all_t = []
catch_all_dist_prior_over_all_mean = []
catch_prior_index = []
#COMPUTE PRIORS
for i in range(20,200):
    catch_prior_index.append(i)
    #MAKE PRIOR FOR MEAN as i
    dist_prior = np.random.poisson(i,N)
    dist_prior_event_probs = pd.Series(dist_prior).value_counts()/pd.Series(dist_prior).value_counts().sum()
    dist_prior_event_probs = dist_prior_event_probs.sort_index()
    catch_all_dist_prior_over_all_mean.append(dist_prior)
    
    #GENERATE OVER ALL T AND COLLECT 
    #THEN COLLECT TO LIST OVER PRIORS (OVER i)
    catch_prior_over_all_t = ([])
    for t in range(1,int(dist_prior_event_probs.index[-1])): 
        dist_1 = bayes.compute_posterior(t, dist_prior_event_probs, N)
        dist_1 = pd.Series(dist_1, index=dist_prior_event_probs.index)
        catch_prior_over_all_t = np.append(catch_prior_over_all_t, bayes.median_of_dist(dist_1))
    catch_all_prior_over_all_t.append(catch_prior_over_all_t)

'''SAVE OBJECTS FOR SIMS AND ANALYSIS'''
with open('catch_all_dist_prior_over_all_mean_out','wb') as filehandle:
    pickle.dump(catch_all_dist_prior_over_all_mean, filehandle)

with open('catch_all_prior_over_all_t_out','wb') as filehandle:
    pickle.dump(catch_all_prior_over_all_t, filehandle)

with open('catch_prior_index_out','wb') as filehandle:
    pickle.dump(catch_prior_index, filehandle)


#EOF
#EOF
#EOF
