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

datadir = '/Users/biocomplexity/Projects/SocioCognitiveModeling/Metaculus_CogModeling/simulations/InputData'
datasavename='Final_Cleaned_Low_R3_R4'

'''
DO LOWEST
ROUND 3, 4
'''
q_list = [9018, 9426]

'''DATES FOR MIN MAX'''
min_r1 = datetime.strptime('20211111', "%Y%m%d")
max_r1 = datetime.strptime('20220202', "%Y%m%d")#offset of -1 from published date accounted for here
min_r2 = datetime.strptime('20211203', "%Y%m%d")
max_r2 = datetime.strptime('20220224', "%Y%m%d")#offset of -1 from published date accounted for here
min_r3 = datetime.strptime('20211224', "%Y%m%d")
max_r3 = datetime.strptime('20220317', "%Y%m%d")#offset of -1 from published date accounted for here
min_r4 = datetime.strptime('20220114', "%Y%m%d")
max_r4 = datetime.strptime('20220407', "%Y%m%d")#offset of -1 from published date accounted for here

'''START DATA CONCAT LOOP'''
catch_miners_maxers_analy = []
counter = 0
for i in q_list:
    print('QUESTION NUMBER: ',i)
    df_S_in = pd.read_csv(f'{datadir}/AllSubs_q{i}.csv')
    df_S_tmp = df_S_in[['user_id','date','0']].copy()
    df_S_tmp.columns = ['user_id','decision_date_str','prediction_date_str']
    df_S_tmp['decision_date'] = df_S_tmp['decision_date_str'].map(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d'))
    df_S_tmp['prediction_date'] = df_S_tmp['prediction_date_str'].map(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d'))
    
    df_use = df_S_tmp.copy()
    df_use = df_use.sort_values(by='decision_date')
    
    begin_wave = datetime.strptime('20211129', "%Y%m%d")
    df_use['begin_wave'] = begin_wave 

    df_use['t'] = df_use.decision_date - df_use.begin_wave
    df_use['t_int'] = df_use['t'].map(lambda x: x.days)

    df_use['prediction'] = df_use.prediction_date - df_use.begin_wave
    df_use['prediction_int'] = df_use['prediction'].map(lambda x: x.days)

    df_use['prediction_horiz'] = df_use.prediction_date - df_use.decision_date
    df_use['prediction_horiz_int'] = df_use['prediction_horiz'].map(lambda x: x.days)
    
    #START MINERS MAXERS VARIABLE CREATION
    if i == 9018: 
        min_rX = min_r3 
        max_rX = max_r3
    if i == 9426:
        min_rX = min_r4 
        max_rX = max_r4
    if i == 8570:
        min_rX = min_r1 
        max_rX = max_r1
    if i == 8799:
        min_rX = min_r2 
        max_rX = max_r2
    if i == 9016:
        min_rX = min_r3 
        max_rX = max_r3
        
    df_use['min_x'] = min_rX
    df_use['min_diff'] = df_use['min_x'] - df_use['prediction_date'] 
    df_use['min_diff_days'] = df_use.min_diff.map(lambda x: x.days)
    df_use['max_x'] = max_rX
    df_use['max_diff'] = df_use['prediction_date'] - df_use['max_x']
    df_use['max_diff_days'] = df_use.max_diff.map(lambda x: x.days)

    '''MAXERS'''
    df_use['maxers'] = 0
    df_use.loc[df_use.max_diff_days==0,'maxers'] = 1
    df_use.maxers.value_counts()

    df_use['max_gooders'] = 0  
    df_use.loc[df_use.max_diff_days<0,'max_gooders'] = 1
    df_use.max_gooders.value_counts()

    '''MINERS'''
    df_use['miners'] = 0
    df_use.loc[df_use.min_diff_days==0,'miners'] = 1
    df_use.miners.value_counts()

    df_use['min_gooders'] = 0
    df_use.loc[df_use.min_diff_days<0,'min_gooders'] = 1
    df_use.min_gooders.value_counts()

    '''IDENTIFICATION OF Ss THAT ARE MINERS OR MAXERS'''
    #EXPORTED TO LIST SEPARATELY FOR EACH QUESTINO
    df_use_gb_user_id_sum = df_use.groupby(by=df_use.user_id).sum() 
    catch_miners_maxers_analy.append([i,
                                      ('right-min',df_use.max_diff_days.min()),
                                      ('right-max',df_use.max_diff_days.max()),
                                      ('left-min',df_use.min_diff_days.min()),
                                      ('left-max',df_use.min_diff_days.max()),
                                      'MINERS Ss',
                                      df_use_gb_user_id_sum.loc[df_use_gb_user_id_sum.miners>0,'miners'],
                                      'MAXERS Ss',
                                      df_use_gb_user_id_sum.loc[df_use_gb_user_id_sum.maxers>0,'maxers']])

    '''CONCAT OVER QUESTIONS'''
    if counter == 0:
        df_use_all = df_use.copy()
    else:
        df_use_all = pd.concat([df_use_all, df_use],ignore_index=True)
    
    counter =+ 1   

#LOOP END

'''DOUBLE CHECK ON MINER MAXERS'''
catch_miners_maxers_analy
#OOOD

'''BEGIN DATA CLEANING'''
#MINERS
df_use_all.miners.value_counts(dropna=False)
#NONE, but put code here for generality
df_use_all = df_use_all[df_use_all.miners==0]
#MAXERS
df_use_all.maxers.value_counts()
#NONE, but put code here for generality
df_use_all = df_use_all[df_use_all.maxers==0]

#T less than the begin_wave, t_int is negative
plt.hist(df_use_all.t_int,bins=100)
df_use_all = df_use_all[df_use_all.t_int>0]

#t greater than prediction (horizon is negative)
plt.hist(df_use_all.prediction_horiz_int,bins=100)
df_use_all = df_use_all[df_use_all.prediction_horiz_int>0]
'''CLEAN FROM HERE ON'''
len(df_use_all)
#LEN IS 195

#TAKE A LOOK FOR SANITY CHECK
df_use_all.columns
df_use_all[['decision_date','prediction_date','t_int','prediction_int','prediction_horiz']]
#SAVE AS PICKLE
df_use_all.to_pickle(f'{datasavename}.pkl')

'''ANALYSIS OF FREQUENCY OF JUDGMENTS PER S'''
user_gb = df_use_all.groupby(by=df_use_all.user_id).count() 
user_gb_out = user_gb.decision_date.sort_values()
user_gb_out.to_csv(f'{datasavename}_S_freqs.csv')

#EOF