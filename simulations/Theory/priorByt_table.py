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
priors_in = '/Users/biocomplexity/Projects/SocioCognitiveModeling/Metaculus_CogModeling/simulations/Priors'
outfile_name = 'sim_out_highest'

'''LOAD PRIORS'''
posteriors = pd.read_pickle(f'{priors_in}/catch_all_prior_over_all_t_out')
prior_means = pd.read_pickle(f'{priors_in}/catch_prior_index_out')

for i in range(0,len(posteriors)):  
    print(prior_means[i],': ',posteriors[i][:prior_means[i]])
    plt.plot(posteriors[i][:prior_means[i]])
    
#PUT INTO NP ARRAY FOR HEAT MAP
catch_array = np.zeros((len(posteriors),len(posteriors)))
for i in range(0,len(posteriors)):
    sub = posteriors[i][:prior_means[i]]
    catch_array[i][0:len(sub)] = sub