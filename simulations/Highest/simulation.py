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

'''LOAD DATA'''
df_use_all = pd.read_pickle(f'{data_in}/Final_Cleaned_High_R1_R3.pkl')

'''END DATA INGEST'''



'''SIMULATIONS'''
'''SIMULATIONS'''
'''SIMULATIONS'''


'''ALL Ss TOGETHER'''
outfile_group = 'all_S'
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
S_all.to_pickle(f'S_all_{outfile_name}_{outfile_group}.csv')



'''SINGLE SUBJECT RUNS'''
df_s_freqs = pd.read_csv(f'{data_in}/Final_Cleaned_High_R1_R3_S_freqs.csv')
s_freq_threshold = 15
s_list = df_s_freqs.loc[df_s_freqs.decision_date>s_freq_threshold,'user_id']

for k in s_list:

    outfile_group = k
    df_S_w2 = df_use_all[df_use_all.user_id==outfile_group].copy()

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
    S_all.to_pickle(f'S_all_{outfile_name}_{outfile_group}.csv')

'''END OF LOOP GO HOME'''















'''DEV'''
'''SINGLE SUBJECT'''
outfile_group = 114156
df_S_w2 = df_use_all[df_use_all.user_id==outfile_group].copy()

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
S_all.to_pickle(f'S_all_{outfile_name}_{outfile_group}.csv')

#EOF