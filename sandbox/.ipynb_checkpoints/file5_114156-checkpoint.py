import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

'''DEF'''
def prob_t_tot(x,dist):
    '''
    x = t_total
    dist is the right distribution and should be a pd.Series
    returns the event probability of t_total as a discrete event, not integrated
    NOTE, the ifelse control should really interpolate but for now it doesnt.
    '''
    if x in dist.index:
        a = dist[x]
    else:
        a = np.float64(0)
    #NOTE, THISFU  
    return a
#EG
#prob_t_tot(34,dist_prior_event_probs)

def n_t_tot(x,dist,n):
    '''
    x=t_total
    dist same as prob_t_tot
    n is N from generating dist.
    returns the N of t_total from the generating distribution
    '''
    if x in dist.index:
        a = dist[x]*n
    else:
        a = np.float64(0)
    #NOTE, THISFU  
    return a
#EG
#n_t_tot(70,dist_prior_event_probs,N)

def median_of_dist(dist):
    '''
    dist is a pd.Series
    returns the median index value which should be
    the value of the poisson
    '''
    return dist.loc[dist.cumsum()>0.5].index[0]
#EG
#median_of_dist(dist_prior_event_probs)

def median_of_dist_p(dist,p):
    '''
    dist is a pd.Series
    returns the median index value which should be
    the value of the poisson
    p = prob value to grab above, 0.5 is median
    '''
    return dist.loc[dist.cumsum()>p].index[0]
#EG
#median_of_dist(dist_prior_event_probs)




def compute_p_t(t,dist):
    '''
    t is for current problem
    dist is dist_prior_event_probs, will be pd series.
    returns a scalar numpy.float64
    '''
    subset = dist_prior_event_probs.iloc[dist_prior_event_probs.index>=t]
    #print('subset',subset)
    one_over_t_total = pd.Series(1/subset.index,index=subset.index) 
    #print('one over t tot',one_over_t_total)
    
    return (subset*one_over_t_total).sum()
#EG
#compute_p_t(5,dist_prior_event_probs)
#compute_p_t(75,dist_prior_event_probs)
'''compute_p_t LOGIpC'''
#dist_prior_event_probs.index
#one_over_ttot = pd.Series(1/dist_prior_event_probs.index,index=dist_prior_event_probs.index)



'''EXAMPLE LOGIC
provide context, 'if 10 days into an epidemic wave (t=10), how many days total (t_total) will be until the peak(or end)'
want to generate the distribution of t_total|t over range of t_total;
t_total drives all the variables, so this function will take in an vector t_total and compute and collect once per
index of t_total
'''

def compute_posterior(t,dist):
    '''
    dist is the ordered probablility lookup table fro each event t_total
    t_total is what we are computing the distribution over
    it is a np.array, ordered
    
    returns np.array, ordered as the 
    NOTE: using index of prior doesn't cover, necessarily, all
    possible values of t_total. Will fix later.
    '''
    #print('t is: ',t)
    vect_t_total = dist.index
    #print('t_total_range is: ',vect_t_total)
    
    catch = np.array([])
    
    for i in vect_t_total:
        #print('t_total is: ',i)
        #prior
        prior = prob_t_tot(i,dist)
        if i < t:
            likelihood = 0
        else: 
            #likelihood = (1/i)*(n_t_tot(i,dist_prior_event_probs,N))
            likelihood = (n_t_tot(i,dist,N)) / ((n_t_tot(i,dist,N))*i)
        p_t = compute_p_t(t,dist)
        likelihood_ratio = likelihood/p_t
        #print('likelihood, p_t, likelihood_ratio, prior, post: ',likelihood, p_t, likelihood_ratio, prior, likelihood_ratio*prior)
        catch = np.append(catch,likelihood_ratio*prior)
        
    return catch

'''
I. GENERATE BASIC DATA STRUCTURE
'''

S_no = 114156

df_S_in = pd.read_csv(f'Sub_{S_no}_BayesData_r1_r4.csv')
df_S_tmp = df_S_in[['user_id','date','0']]
df_S_tmp.columns = ['user_id','decision_date_str','prediction_date_str']
df_S_tmp['decision_date'] = df_S_tmp['decision_date_str'].map(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d'))
df_S_tmp['prediction_date'] = df_S_tmp['prediction_date_str'].map(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d'))
df_S = df_S_tmp.copy()
df_S['prediction_duration'] = df_S.prediction_date - df_S.decision_date
df_S['prediction_duration_int'] = df_S['prediction_duration'].map(lambda x: x.days)

#WAVE BEGIN TIMES, SO CAN FIX t
begin_w1 = datetime.strptime('20210623', "%Y%m%d")
begin_w2 = datetime.strptime('20211129', "%Y%m%d")
df_S['begin_w1'] = begin_w1
df_S['begin_w2'] = begin_w2
'''ESTIMATE OF PRIOR FROM W1'''
begin_w1 - begin_w2
#159 days BUT THIS IS FIXED ON A FIXED ACROSS S PRIOR

df_S['t_w2'] = df_S.decision_date - df_S.begin_w2
df_S['t_w2_int'] = df_S['t_w2'].map(lambda x: x.days)

df_S['prediction_w2'] = df_S.t_w2_int + df_S.prediction_duration_int
#NEXT SHOULD BE EQUVALENT TO ABOVE
df_S['prediction_w2_test'] = df_S.prediction_date - df_S.begin_w2
#TEST GOOD




'''
II.  DECIDE ON CUT POINT FOR W2
'''

'''
FIRST SHOWS PREDICTION
SECOND SHOWS PREDICTED DATE
'''
df_S.prediction_duration_int.plot()
df_S.prediction_w2.plot()

'''THIS IS AN USEFUL DATA STRUCTURE FOR DECIDTING WHEN W2 DECISIONS ARE MADE'''
df_S[['t_w2_int','prediction_w2','prediction_duration_int','decision_date','prediction_date']]

'''FURTHER EXPORE OF MIN JUDGMENTS
1. WERE THEY ALL MINERS
2. How INTERPRET NEGATIVES FROM '''
df_S[df_S.prediction_duration_int<0]
df_S[['decision_date','prediction_duration_int']]

'''
ONLY INCLUDE DECISION DATES AFTER THE FOLLOWING CRITERIA
1. ON OR AFTER ESTIMATED t_0 DATE
2. May DELETE OR RE-INTERPRET NEGATIVEs 
'''
tmp_t = pd.to_datetime(datetime.strptime('2021-12-25T17:00:00','%Y-%m-%dT%H:%M:%S')) 
df_S_w2 = df_S[df_S.decision_date > tmp_t ]




'''
III.  GENERATE PRIORS
'''

#FOR PRIOR
N = 10

'''COMPUTE PRIORS'''
catch_all_prior_over_all_t = []
catch_prior_index = []
#COMPUTE PRIORS
for i in range(60,180):
    catch_prior_index.append(i)
    #MAKE PRIOR FOR MEAN as i
    dist_prior = np.random.poisson(i,N)
    dist_prior_event_probs = pd.Series(dist_prior).value_counts()/pd.Series(dist_prior).value_counts().sum()
    dist_prior_event_probs = dist_prior_event_probs.sort_index()
    
    #GENERATE OVER ALL T AND COLLECT 
    #THEN COLLECT TO LIST OVER PRIORS (OVER i)
    catch_prior_over_all_t = ([])
    for t in range(1,int(dist_prior_event_probs.index[-1])): 
        dist_1 = compute_posterior(t, dist_prior_event_probs)
        dist_1 = pd.Series(dist_1, index=dist_prior_event_probs.index)
        catch_prior_over_all_t = np.append(catch_prior_over_all_t, median_of_dist(dist_1))
    catch_all_prior_over_all_t.append(catch_prior_over_all_t)



    
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
plot_this.to_csv(f'plot_this_{S_no}.csv',header=['best_prior'])





#EOF
#EOF
#EOF
