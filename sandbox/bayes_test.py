import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import pickle

from imp import reload
import sys
sys.path.insert(1,'/Users/biocomplexity/Projects/SocioCognitiveModeling/Metaculus_CogModeling/bayes_pack_testing')
import bayes

#reload(bayes)


'''
THE DATA AND MODELING FOR S-109742 HIGHEST
WILL BE USED TO TEST, TAKEN FROM file7_109742.py
but modified as needed below for testing purposes.
'''

'''GENERATE BASIC DATA STRUCTURE'''
S_no = 109742

df_S_in = pd.read_csv(f'Sub_{S_no}_BayesData_r1_r4_highest.csv')
df_S_tmp = df_S_in[['user_id','date','0']]
df_S_tmp.columns = ['user_id','decision_date_str','prediction_date_str']
df_S_tmp['decision_date'] = df_S_tmp['decision_date_str'].map(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d'))
df_S_tmp['prediction_date'] = df_S_tmp['prediction_date_str'].map(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d'))
df_S = df_S_tmp.copy()


#df_S['prediction_duration'] = df_S.prediction_date - df_S.decision_date
#df_S['prediction_duration_int'] = df_S['prediction_duration'].map(lambda x: x.days)

#WAVE BEGIN TIMES, SO CAN FIX t
#begin_w1 = datetime.strptime('20210623', "%Y%m%d")
begin_w2 = datetime.strptime('20211129', "%Y%m%d")
#df_S['begin_w1'] = begin_w1
df_S['begin_w2'] = begin_w2
'''ESTIMATE OF PRIOR FROM W1'''
#begin_w1 - begin_w2
#159 days BUT THIS IS FIXED ON A FIXED ACROSS S PRIOR

df_S['t_w2'] = df_S.decision_date - df_S.begin_w2
df_S['t_w2_int'] = df_S['t_w2'].map(lambda x: x.days)

#df_S['prediction_w2'] = df_S.t_w2_int + df_S.prediction_duration_int
#NEXT SHOULD BE EQUVALENT TO ABOVE
df_S['prediction_w2_test'] = df_S.prediction_date - df_S.begin_w2
#TEST GOOD

df_S_w2 = df_S.copy()



'''
III.  GENERATE PRIORS
'''

#FOR PRIOR
N = 10

'''COMPUTE PRIORS'''
catch_all_prior_over_all_t = []
catch_prior_index = []
#COMPUTE PRIORS
for i in range(20,21):
    catch_prior_index.append(i)
    #MAKE PRIOR FOR MEAN as i
    dist_prior = np.random.poisson(i,N)
    dist_prior_event_probs = pd.Series(dist_prior).value_counts()/pd.Series(dist_prior).value_counts().sum()
    dist_prior_event_probs = dist_prior_event_probs.sort_index()
    
    #GENERATE OVER ALL T AND COLLECT 
    #THEN COLLECT TO LIST OVER PRIORS (OVER i)
    catch_prior_over_all_t = ([])
    for t in range(1,int(dist_prior_event_probs.index[-1])):
        
        dist_1 = bayes.compute_posterior(t, dist_prior_event_probs,N)
        dist_1 = pd.Series(dist_1, index=dist_prior_event_probs.index)
        catch_prior_over_all_t = np.append(catch_prior_over_all_t, bayes.median_of_dist(dist_1))
    catch_all_prior_over_all_t.append(catch_prior_over_all_t)

for t in range(1,int(dist_prior_event_probs.index[-1])): print(t)

    
'''
IV.  GENERATE BEST PRIORS FOR W2
'''    
catch_all_t_over_t_over_p = []
catch_all_optimal_pred_over_t_over_p = []
catch_all_human_pred_over_t_over_p = []
catch_all_error_over_t_over_p = []
catch_all_date_over_t_over_p = []

for j in catch_all_prior_over_all_t: #LOOP OVER PRIORS
    
    print('NEW PRIOR')
    print('NEW PRIOR')
    catch_all_t_over_t = ([])
    catch_all_optimal_pred_over_t = ([])
    catch_all_human_pred_over_t = ([])
    catch_all_error_over_t = ([])
    catch_all_date_over_t = ([])

    for i in range(0,len(df_S_w2)):#CAPTURE HUMAN DATA T AND PRED
        t = df_S_w2.t_w2_int.iloc[i] #P
        print('t',t)
        catch_all_t_over_t = np.append(catch_all_t_over_t,t)
        #optimal_pred = catch_all_prior_over_all_t[0][t-1]#index zero is t=1
        #print('LEN of J',len(j))
        if t-1<len(j):
            optimal_pred = j[t-1]#index zero is t=1
        else:
            print('T-1 OUT OF RANGE OF Ts in PRIOR')
            optimal_pred = j[-1]
        catch_all_optimal_pred_over_t = np.append(catch_all_optimal_pred_over_t,optimal_pred)
        print('optimal pred',optimal_pred)
        human_pred = df_S_w2.prediction_w2.iloc[i]
        catch_all_human_pred_over_t = np.append(catch_all_human_pred_over_t,human_pred)
        print('human pred', human_pred)
        error = human_pred - optimal_pred
        print('error',error)
        catch_all_error_over_t = np.append(catch_all_error_over_t,error)
        catch_all_date_over_t = np.append(catch_all_date_over_t,df_S_w2.decision_date.iloc[i])
        print('decision_date',df_S_w2.decision_date.iloc[i])
    
    catch_all_t_over_t_over_p.append(catch_all_t_over_t)
    catch_all_optimal_pred_over_t_over_p.append(catch_all_optimal_pred_over_t)
    catch_all_human_pred_over_t_over_p.append(catch_all_human_pred_over_t)
    catch_all_error_over_t_over_p.append(catch_all_error_over_t)
    catch_all_date_over_t_over_p.append(catch_all_date_over_t)


'''PICK BEST PRIOR'''
for i in catch_all_error_over_t_over_p: plt.plot(i)
#S HAS TWO REGIMES (one with constant decreasing)
#SEE: 
df_error = pd.DataFrame(catch_all_error_over_t_over_p).T
#df_error['decision_date'] = df_S_w2.decision_date.reset_index(drop=True)
df_error.index = df_S_w2.decision_date.reset_index(drop=True)
df_error.columns = catch_prior_index
#TESTS BLOW
df_error.abs().idxmin(axis=1).plot()
df_error.abs().idxmin(axis=1)
df_error.abs().min(axis=1)
df_error.min(axis=1)
#MAKE THIS FOR MAIN ANALYSIS STRUCTUR
plot_this = df_error.abs().idxmin(axis=1)
plot_this.to_csv(f'plot_this_{S_no}_highest.csv',header=['best_prior'])





#EOF