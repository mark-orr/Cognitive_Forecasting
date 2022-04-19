import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def find_optimal(t_i,medians_for_ts):
    '''
    medians_for_ts expects an np.array, for which each array index = t_i in the model and the 
    array value is the median optimal decision given a prior e.g., catch_all_prior_over_all_t[0] 

    t_i expects something that reflects the time into the process at which the decision was made
    '''
    if (t_i>0) & (t_i<=len(medians_for_ts)):#t-time is medians_for_ts[medians_for_ts.index+1]
        print('T IS WITHIN PRIOR,  GOOOD,b_fit')
        optimal_pred = medians_for_ts[t_i-1]
    else:
        print('T IS GREATER THAN Ts in PRIOR,b_fit')
        optimal_pred = medians_for_ts[-1]

    return optimal_pred

#EOF
