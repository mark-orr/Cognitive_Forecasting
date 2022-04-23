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


'''
THE DATA AND MODELING FOR S-109742 HIGHEST
WILL BE USED TO TEST, TAKEN FROM file7_109742.py
but modified as needed below for testing purposes.
'''

'''GENERATE BASIC DATA STRUCTURE'''
S_no = 112076

df_S_in = pd.read_csv(f'Sub_{S_no}_BayesData_r1_r4_highest.csv')
df_S_tmp = df_S_in[['user_id','date','0']]
df_S_tmp.columns = ['user_id','decision_date_str','prediction_date_str']
df_S_tmp['decision_date'] = df_S_tmp['decision_date_str'].map(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d'))
df_S_tmp['prediction_date'] = df_S_tmp['prediction_date_str'].map(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d'))
df_S = df_S_tmp.copy()

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
df_S['prediction_w2'] = df_S.prediction_date - df_S.begin_w2
df_S['prediction_w2_int'] = df_S['prediction_w2'].map(lambda x: x.days)
#TEST GOOD

df_S['prediction_duration'] = df_S.prediction_date - df_S.decision_date
df_S['prediction_duration_int'] = df_S['prediction_duration'].map(lambda x: x.days)


df_S_w2 = df_S.copy()

'''
SHOULD ADD SOME ADDITIONAL CLEANING STEPS
1. remove t < 0
2. remove pred duration < 0
'''
df_S_w2 = df_S_w2[df_S_w2.t_w2_int > 0]
df_S_w2 = df_S_w2[df_S_w2.prediction_duration_int>0]



'''
FUN TEST

'''
index_for_all = pd.date_range('2022-01-01','2022-02-14')
S_0 = pd.Series(pd.date_range('2022-01-01','2022-02-14'),name='decision_date')
S_1 = pd.Series(range(1,1+len(index_for_all)),name='t_w2_int')
S_2 = pd.Series(np.repeat(45,len(index_for_all)),name='prediction_w2_int')
S_3 = pd.Series(45-S_1,name='prediction_duration_int')

df_S_w2 = pd.concat([S_0,S_1,S_2,S_3],axis=1)


'''
III.  GENERATE PRIORS
'''

#FOR PRIOR
N = 1000

'''COMPUTE PRIORS'''
catch_all_prior_over_all_t = []
catch_prior_index = []
#COMPUTE PRIORS
for i in range(38,46):
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
catch_tmp_optimal_over_p = []
catch_all_human_pred_over_t_over_p = []
catch_all_error_over_t_over_p = []
catch_all_date_over_t_over_p = []
catch_all_p_dur_over_t_over_p = []

for j in catch_all_prior_over_all_t: #LOOP OVER PRIORS
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
    #for i in range(0,3):#CAPTURE HUMAN DATA T AND PRED
        t = df_S_w2.t_w2_int.iloc[i] #P
        
        p_dur = df_S_w2.prediction_duration_int.iloc[i]
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
        
        human_pred = df_S_w2.prediction_w2_int.iloc[i]
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
#for i in catch_all_error_over_t_over_p: plt.plot(i)
#for i in catch_all_p_dur_over_t_over_p: plt.plot(i)
#for i in catch_all_optimal_pred_over_t_over_p: plt.plot(i)
'''PLOT THESE TOGETHER FOR A NICE LOOK AND SANITY CHECK'''
#S HAS TWO REGIMES (one with constant decreasing)
#SEE: 
df_error = pd.DataFrame(catch_all_error_over_t_over_p).T
#df_error['decision_date'] = df_S_w2.decision_date.reset_index(drop=True)
df_error.index = df_S_w2.decision_date.reset_index(drop=True)
df_error.columns = catch_prior_index
#TESTS BLOW
#df_error.abs().idxmin(axis=1).plot()
#df_error.abs().idxmin(axis=1)
#df_error.abs().min(axis=1)
#df_error.min(axis=1)
#MAKE THIS FOR MAIN ANALYSIS STRUCTUR
plot_this = df_error.abs().idxmin(axis=1)
#plot_this.to_csv(f'plot_this_{S_no}_highest.csv',header=['best_prior'])

#MAKE THIS FOR SANITY CHECK
S_ts = pd.Series(catch_all_t_over_t,index=df_S_w2.decision_date.reset_index(drop=True),name='t')
S_hp = pd.Series(catch_all_human_pred_over_t,index=df_S_w2.decision_date.reset_index(drop=True),name='hum')
S_pd = pd.Series(catch_all_p_dur_over_t,index=df_S_w2.decision_date.reset_index(drop=True),name='p_dur')
S_er = df_error.abs().min(axis=1)
plot_this.name='prior'
S_er.name='err'

S_all = pd.concat([S_ts, S_hp, S_pd, plot_this, S_er], axis=1)
S_all.plot()


#EOF