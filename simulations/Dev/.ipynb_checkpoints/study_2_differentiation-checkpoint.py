import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import pickle
from scipy import stats
import matplotlib.font_manager as fm   
from pylab import cm
import matplotlib as mpl

from imp import reload
import sys
sys.path.insert(1,'/Users/biocomplexity/Projects/SocioCognitiveModeling/Metaculus_CogModeling/bayes_pack_testing')
import bayes
import b_fit

reload(bayes)

import changefinder
def findChangePoints(ts, r, order, smooth):
    '''
       r: Discounting rate
       order: AR model order
       smooth: smoothing window size T
    '''
    cf = changefinder.ChangeFinder(r=r, order=order, smooth=smooth)
    ts_score = [cf.update(p) for p in ts]
    plt.figure(figsize=(16,4))
    plt.plot(ts)
    plt.figure(figsize=(16,4))
    plt.plot(ts_score, color='red')
    return(ts_score)

cases_data_in = '/Users/biocomplexity/Projects/SocioCognitiveModeling/Metaculus_CogModeling/simulations/InputData'

'''CODE STRUCTURE:
+ data block

+ raw data --> poly fit --> f(x)' and f(x)''
for both human and vdh raw and tried 
human smoothed inplaced of poly fit.

+ raw data --> perc. ch. --> 
'''


'''
%%%%%%%%%%
DATA BLOCK
'''

'''EPI DATA'''
vdh = pd.read_csv(f'{cases_data_in}/VDH-COVID-19-PublicUseDataset-Cases.csv',parse_dates=['report_date'])

vdh = vdh.groupby(['report_date'])['total_cases'].sum().diff().rolling(window=7).mean().round()
vdh_use = vdh['11-30-2021':'01-14-2022'].copy()
vdh_use.name = 'use'

'''ADD HUMAN DATA'''
data_in = '/Users/biocomplexity/Projects/SocioCognitiveModeling/Metaculus_CogModeling/simulations/Preprocessing'
infile_name = 'sim_out_highest'

df_s_freqs = pd.read_csv(f'{data_in}/Final_Cleaned_High_R1_R3_S_freqs.csv')
s_freq_threshold = 15
s_S = df_s_freqs.loc[df_s_freqs.decision_date>s_freq_threshold,'user_id']
s_list = s_S.to_list()
group_list = ['all_s'] + s_list


'''PUT ALL DATA FOR EACH GROUP IN LIST'''
catch_groups = []
for i in group_list:
    '''ANALYZE All Ss'''
    infile_group = i

    '''GRAB DATA'''
    S_all = pd.read_pickle(f'S_all_{infile_name}_{infile_group}.csv')
    
    catch_groups.append(S_all)


gp_no = group_list[0]
i = catch_groups[0]
'''aVE OVER DAY'''
grouped = i.groupby(level=0)

hum_use = grouped.hum.mean()


'''
%%%%%%%%%%%%%%%%%%%%%
DIFFERENTIATION BLOCK
USING SIMPLE POLY FIT
'''

'''VDH'''
#PROCESS RAW DATA FOR DIFFERENTIATION
x_poly = np.arange(len(vdh_use))+1
vdh_poly = np.polyfit(x_poly, vdh_use, deg=6)
vdh_poly_values = np.polyval(vdh_poly, x_poly)
vdh_poly_values_S = pd.Series(vdh_poly_values,index=vdh_use.index)

#FIRST DERIVATIVE
v_x1 = np.diff(vdh_poly_values)
v_x1_S = pd.Series(v_x1,index=vdh_use.index[1:])

#SECOND DERIVATIVE
v_x2 = np.diff(v_x1)
v_x2 = np.insert(v_x2,0,0,axis=0)#NOW SAME LEN AS v_x1 and indexes matched in time.
v_x2_S = pd.Series(v_x2,index=vdh_use.index[1:])

#SANITY PLOTS
plt.plot(v_x1_S); plt.plot(v_x2_S); plt.plot(vdh_poly_values_S*0.1); plt.plot(vdh_use[2:]*0.1); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)
#plt.savefig('vdh_tmp1.png',dpi=200)

#plt.plot(v_x1_S); plt.plot(vdh_use[2:]*0.1); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)
#plt.plot(v_x2_S);  plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)



'''HUMAN'''
#PROCESS RAW DATA FOR DIFFERENTIATION
x_poly = np.arange(len(hum_use))+1
hum_poly = np.polyfit(x_poly, hum_use, deg=7)
hum_poly_values = np.polyval(hum_poly, x_poly)
hum_poly_values_S = pd.Series(hum_poly_values,index=hum_use.index)

#FIRST DERIVATIVE
h_x1 = np.diff(hum_poly_values)
h_x1_S = pd.Series(h_x1,index=hum_use.index[1:])

#SECOND DERIVATIVE
h_x2 = np.diff(h_x1)
h_x2 = np.insert(h_x2,0,0,axis=0)#NOW SAME LEN AS v_x1 and indexes matched in time.
h_x2_S = pd.Series(h_x2,index=hum_use.index[1:])

#SANITY PLOTS
plt.plot(h_x1_S); plt.plot(h_x2_S); plt.plot(hum_poly_values_S*0.1); plt.plot(hum_use[2:]*0.1); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)
#plt.savefig('hum_tmp1_2.png',dpi=200)

#plt.plot(h_x1_S); plt.plot(h_x2_S); plt.plot(hum_use[2:]*1); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)
#plt.plot(h_x1_S); plt.plot(hum_use[2:]*0.1); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)
#plt.plot(h_x1); plt.plot(h_x2); plt.plot(hum_poly_values); plt.plot(np.array(hum_use[1:]))
#plt.plot(h_x2_S);  plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)


'''
PLOTTING EPI AND HUMAN TOGETHER
'''
plt.plot(v_x1_S); plt.plot(v_x2_S); plt.plot(vdh_poly_values_S*0.1); plt.plot(vdh_use[2:]*0.1); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)
plt.plot(h_x1_S); plt.plot(h_x2_S); plt.plot(hum_poly_values_S*0.1); plt.plot(hum_use[2:]*0.1); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)

'''
USABEL PLOT FOR MANUSCRIPT
EPI AND HUMAN RAW DATA TOGETHER
'''
'''NEED CHANGE POINTS FOR PLOT'''
v_ts1 = np.array(vdh_use.values)
v_ts_score1 = findChangePoints(v_ts1, r = 0.01, order = 3, smooth = 5)
v_ts_change_loc1 = pd.Series(v_ts_score1).nlargest(6)
v_ts_change_loc1 = v_ts_change_loc1.index
v_ts_change_loc1
#TAKE FIRST IN A BLOCK, 11, 20, 38

h_ts1 = np.array(hum_use.values)
h_ts_score1 = findChangePoints(h_ts1, r = 0.01, order = 3, smooth = 5)
h_ts_change_loc1 = pd.Series(h_ts_score1).nlargest(6)
h_ts_change_loc1 = h_ts_change_loc1.index
h_ts_change_loc1

'''NOW MAKE GRAPH'''
t = vdh_use.index
data1 = vdh_use
data2 = hum_use
data3 = vdh_poly_values_S
data4 = hum_poly_values_S
#PLOT
fig, ax1 = plt.subplots()
color = 'black'
ax1.set_xlabel('Date')
ax1.set_ylabel('Cases', color=color)
ax1.plot(t, data1, color=color)
ax1.plot(t, data3, color=color)
ax1.scatter(t,data1, color=color,marker='+',label='Cases',s=100)
ax1.tick_params(axis='y', labelcolor=color)
ax1.axvline(x=t[11],c='black',dashes=(2,2,2,2),alpha=.4)
ax1.axvline(x=t[20],c='black',dashes=(2,2,2,2),alpha=.4)
ax1.axvline(x=t[38],c='black',dashes=(2,2,2,2),alpha=.4)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'black'
ax2.set_ylabel('Days', color=color)  # we already handled the x-label with ax1
ax2.plot(t, data2, color=color)
ax2.plot(t, data4, color=color,dashes=(2,2,2,2))
ax2.scatter(t,data2, color=color,marker='s',label='human t_pred (Days)')
ax2.tick_params(axis='y', labelcolor=color)
#ax2.axvline(x=t[11],c='black',alpha=.4)
#ax2.axvline(x=t[26],c='black',dashes=(4,2,4,2),alpha=.4)
#ax2.axvline(x=t[35],c='black',dashes=(4,2,4,2),alpha=.4)
ax1.tick_params(axis='x', labelsize=7)

fig.legend(bbox_to_anchor=(.85, .3), frameon=False, fontsize=10)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()



'''FIRST DIFFERENTIAL'''
'''NOW MAKE GRAPH'''
t = vdh_use.index
data1 = v_x1_S
data2 = h_x1_S

#PLOT
fig, ax1 = plt.subplots()
color = 'black'

ax1.set_xlabel('Date')
ax1.set_ylabel('Cases', color=color)
ax1.plot(t[1:], data1, color=color,label='Cases')
ax1.tick_params(axis='y', labelcolor=color)
#ax1.axvline(x=t[11],c='black',dashes=(2,2,2,2),alpha=.4)
#ax1.axvline(x=t[20],c='black',dashes=(2,2,2,2),alpha=.4)
#ax1.axvline(x=t[38],c='black',dashes=(2,2,2,2),alpha=.4)
ax1.axhline(y=0,color=color,alpha=.4)
ax1.set_ylim(-262, 890)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'black'
ax2.set_ylabel('Days', color=color)  # we already handled the x-label with ax1
ax2.plot(t[1:], data2, color=color,dashes=(2,2,2,2),label='human t_pred (Days)')
ax2.tick_params(axis='y', labelcolor=color)
ax2.axhline(y=0,color=color,alpha=.4,dashes=(2,2,2,2))
#ax.axvline(x=t[11],c='black',alpha=.4)
#ax2.axvline(x=t[26],c='black',dashes=(4,2,4,2),alpha=.4)
#ax2.axvline(x=t[35],c='black',dashes=(4,2,4,2),alpha=.4)
ax2.tick_params(axis='x', labelsize=7)

fig.legend(bbox_to_anchor=(.55, .95), frameon=False, fontsize=10)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


'''SECOND DIFFERENTIAL'''
'''NOW MAKE GRAPH'''
t = vdh_use.index
data1 = v_x2_S
data2 = h_x2_S

#PLOT
fig, ax1 = plt.subplots()
color = 'black'
ax1.set_xlabel('Date')
ax1.set_ylabel('Cases', color=color)
ax1.plot(t[1:], data1, color=color,label='Cases')
ax1.tick_params(axis='y', labelcolor=color)
#ax1.axvline(x=t[11],c='black',dashes=(2,2,2,2),alpha=.4)
#ax1.axvline(x=t[20],c='black',dashes=(2,2,2,2),alpha=.4)
#ax1.axvline(x=t[38],c='black',dashes=(2,2,2,2),alpha=.4)
ax1.axhline(y=0,color=color,alpha=.4)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'black'
ax2.set_ylabel('Days', color=color)  # we already handled the x-label with ax1
ax2.plot(t[1:], data2, color=color,dashes=(2,2,2,2),label='human t_pred (Days)')
ax2.tick_params(axis='y', labelcolor=color)
ax2.axhline(y=0,color=color,alpha=.4,dashes=(2,2,2,2))
#ax2.axvline(x=t[11],c='black',alpha=.4)
#ax2.axvline(x=t[26],c='black',dashes=(4,2,4,2),alpha=.4)
#ax2.axvline(x=t[35],c='black',dashes=(4,2,4,2),alpha=.4)
ax2.set_ylim(-0.7,1.33)

ax1.tick_params(axis='x', labelsize=7)

fig.legend(bbox_to_anchor=(.55, .95), frameon=False, fontsize=10)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()



fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.set_xlabel('Date')
ax1.set_ylabel('Cases', color=color)
ax1.plot(t[1:], data1, color=color,label='Cases')
ax1.tick_params(axis='y', labelcolor=color)
#ax1.axvline(x=t[11],c='black',dashes=(2,2,2,2),alpha=.4)
#ax1.axvline(x=t[20],c='black',dashes=(2,2,2,2),alpha=.4)
#ax1.axvline(x=t[38],c='black',dashes=(2,2,2,2),alpha=.4)
ax1.axhline(y=0,color=color,alpha=.4)

ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'black'
ax3.set_ylabel('Days', color=color)  # we already handled the x-label with ax1
ax3.plot(t[1:], data2, color=color,dashes=(2,2,2,2),label='human t_pred (Days)')
ax3.tick_params(axis='y', labelcolor=color)
ax3.axhline(y=0,color=color,alpha=.4,dashes=(2,2,2,2))
#ax2.axvline(x=t[11],c='black',alpha=.4)
#ax2.axvline(x=t[26],c='black',dashes=(4,2,4,2),alpha=.4)
#ax2.axvline(x=t[35],c='black',dashes=(4,2,4,2),alpha=.4)
ax3.set_ylim(-0.7,1.33)








h_scale = 5
v_scale = 0.01
plt.plot(vdh_use[2:]*v_scale); 
plt.plot(hum_use[2:]*h_scale); 
plt.plot(vdh_poly_values_S*v_scale); 
plt.plot(hum_poly_values_S*h_scale)
plt.avhline(y=30, c='r',dashes=(2,2,2,2),linewidth=2)

h_scale = 5
v_scale = 0.01
plt.plot(v_x1_S*v_scale); plt.plot(h_x1_S*h_scale)
plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)

h_scale = 5
v_scale = 0.01
plt.plot(v_x2_S*v_scale); plt.plot(h_x2_S*h_scale)
plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)

'''USEFUL PHASE PLOTS'''

#NO ARROWS
x = vdh_poly_values
y = hum_poly_values 

fig, ax = plt.subplots()
ax.scatter(x,y)
ax.plot(x,y)
alpha_correction = 1/len(x)
'''TRY COMET TAIL'''
for i in range(0,len(x)): plt.scatter(x[i],y[i],alpha=alpha_correction*i,color='black',marker='+'); plt.plot(x,y,linewidth=0.2,alpha=0.5)
plt.savefig('tmp_phase_space.png',dpi=300)




'''TESTING SIMPLE CHANGE POINTS'''

v_ts1 = np.array(vdh_use.values)
v_ts_score1 = findChangePoints(v_ts1, r = 0.01, order = 3, smooth = 5)
v_ts_change_loc1 = pd.Series(v_ts_score1).nlargest(5)
v_ts_change_loc1 = v_ts_change_loc1.index
v_ts_change_loc1


h_ts1 = np.array(hum_use.values)
h_ts_score1 = findChangePoints(h_ts1, r = 0.01, order = 3, smooth = 5)
h_ts_change_loc1 = pd.Series(h_ts_score1).nlargest(5)
h_ts_change_loc1 = h_ts_change_loc1.index
h_ts_change_loc1


'''JUST FOR FUN, LOOK AT PC IN DIFF'''
#h_u_pc_r = hum_poly_values_S.rolling(4).mean().pct_change()
#v_u_pc_r = vdh_poly_values_S.rolling(4).mean().pct_change()
h_u_pc_r = hum_poly_values_S.pct_change()
v_u_pc_r = vdh_poly_values_S.pct_change()

for i in range(4,9):
    v_u_h_u_r_corr = v_u_pc_r.rolling(i).corr(h_u_pc_r)
    plt.plot(v_u_h_u_r_corr,label=i)
    plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=1)
    plt.axvline(x=datetime.strptime('2021-12-03','%Y-%m-%d'),c='black',dashes=(2,2,2,2),linewidth=2,alpha=0.3)
    plt.axvline(x=datetime.strptime('2021-12-24','%Y-%m-%d'),c='black',dashes=(2,2,2,2),linewidth=2,alpha=0.3)
    plt.axvline(x=datetime.strptime('2022-01-14','%Y-%m-%d'),c='black',dashes=(2,2,2,2),linewidth=2,alpha=0.3)
    plt.legend()


'''JUST FOR FUN, LOOK AT PC IN DIFF'''
h_u_pc_r = h_x1_S.pct_change()
v_u_pc_r = v_x1_S.pct_change()

for i in range(4,9):
    v_u_h_u_r_corr = v_u_pc_r.rolling(i).corr(h_u_pc_r)
    plt.plot(v_u_h_u_r_corr,label=i)
    plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=1)
    plt.axvline(x=datetime.strptime('2021-12-03','%Y-%m-%d'),c='black',dashes=(2,2,2,2),linewidth=2,alpha=0.3)
    plt.axvline(x=datetime.strptime('2021-12-24','%Y-%m-%d'),c='black',dashes=(2,2,2,2),linewidth=2,alpha=0.3)
    plt.axvline(x=datetime.strptime('2022-01-14','%Y-%m-%d'),c='black',dashes=(2,2,2,2),linewidth=2,alpha=0.3)
    plt.legend()
    
    

'''LAG-N CROSS CORR'''
def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 
    
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))

'''FOR RAW TIME SERIES'''
#d1 = pd.Series(np.array(hum_use))
#d2 = pd.Series(np.array(vdh_use))
'''PERCENT CHANGE TIME SERIES'''
d1 = v_x2_S
d2 = h_x2_S

rs = [crosscorr(d1,d2, lag) for lag in range(-10,10)]
offset = np.floor(len(rs)/2)-np.argmax(rs)
f,ax=plt.subplots(figsize=(14,3))
ax.plot(rs)
ax.axvline(np.ceil(len(rs)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(rs),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Offset = {offset} frames\nS1 leads <> S2 leads',ylim=[-1,1],xlim=[0,41], xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 10, 20])
ax.set_xticklabels([-10, 0, 10]);
plt.legend()


'''PHASE SYCHRONY'''

from scipy.signal import hilbert, butter, filtfilt
from scipy.fftpack import fft,fftfreq,rfft,irfft,ifft
import seaborn as sns

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

lowcut  = .1
highcut = .9
fs = 5
order = 1

#d1 = pd.Series(np.array(hum_use.pct_change())).interpolate().values
#d2 = pd.Series(np.array(vdh_use.pct_change())).interpolate().values
#d1 = hum_use.interpolate().values
#d2 = vdh_use.interpolate().values
d1 = v_x1
d2 = h_x1
y1 = butter_bandpass_filter(d1,lowcut=lowcut,highcut=highcut,fs=fs,order=order)
y2 = butter_bandpass_filter(d2,lowcut=lowcut,highcut=highcut,fs=fs,order=order)
plt.plot(y1)
plt.plot(y2)

al1 = np.angle(hilbert(y1),deg=False)
al2 = np.angle(hilbert(y2),deg=False)
phase_synchrony = 1-np.sin(np.abs(al1-al2)/2)
N = len(al1)


# Plot results
f,ax = plt.subplots(3,1,figsize=(14,7),sharex=True)
ax[0].plot(y1,color='r',label='y1')
ax[0].plot(y2,color='b',label='y2')
ax[0].legend(bbox_to_anchor=(0., 1.02, 1., .102),ncol=2)
ax[0].set(xlim=[0,N], title='Filtered Timeseries Data')
ax[1].plot(al1,color='r')
ax[1].plot(al2,color='b')
ax[1].set(ylabel='Angle',title='Angle at each Timepoint',xlim=[0,N])
phase_synchrony = 1-np.sin(np.abs(al1-al2)/2)
ax[2].plot(phase_synchrony)
ax[2].set(ylim=[0,1.1],xlim=[0,N],title='Instantaneous Phase Synchrony',xlabel='Time',ylabel='Phase Synchrony')
plt.tight_layout()
plt.show()


'''
HOME SPUN
PHASE SPACE ANALYSIS
'''
#USE df_b for phase space.
df_b.columns = ['epi_vel','hum_acc']

#MAKE X and Y
S_x = v_x1.copy()
S_x = np.insert(S_x,0,0,axis=0) #PAD FIRST
S_x = S_x[:-1].copy() #REMOVE LAST
S_x = pd.Series(S_x,index=df_b.index)

S_y = h_x2.copy()
S_y = np.insert(S_y,0,0,axis=0) #PAD FIRST
S_y = S_y[:-1].copy() #REMOVE LAST
S_y = pd.Series(S_y,index=df_b.index)

df_b['S_x'] = S_x
df_b['S_y'] = S_y

thing = v_x1.copy()
catch_arr = np.array([])
for i in range(1,len(thing)):
    catch_arr = np.append(catch_arr,thing[i]-thing[i-1])
S_dx = catch_arr.copy()
S_dx = np.insert(S_dx,0,0,axis=0)
df_b['S_dx'] = S_dx

thing = h_x2.copy()
catch_arr = np.array([])
for i in range(1,len(thing)):
    catch_arr = np.append(catch_arr,thing[i]-thing[i-1])
S_dy = catch_arr.copy()
S_dy = np.insert(S_dy,0,0,axis=0)
df_b['S_dy'] = S_dy




'''THIS IS THE GOOD ONE, BUT NOT QUITE PUB QUALITY'''
x_min = min(df_b.S_x)-50
x_max = max(df_b.S_x)+50
y_min = min(df_b.S_y)+4#SPECIAL
y_max = max(df_b.S_y)+1
             

#ARROWS
x = df_b.S_x.iloc[2:10].values
y = df_b.S_y.iloc[2:10].values

u = np.diff(x)
v = np.diff(y)
pos_x = x[:-1] + u/2
pos_y = y[:-1] + v/2
norm = np.sqrt(u**2+v**2) 

fig, ax = plt.subplots(); ax.plot(x,y, marker="o"); ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid"); ax.set_xlim([x_min,x_max]); ax.set_ylim([y_min,y_max])
plt.savefig('Time_Steps_2-10.png',dpi=200)

#ARROWS
x = df_b.S_x.iloc[9:20].values
y = df_b.S_y.iloc[9:20].values

u = np.diff(x)
v = np.diff(y)
pos_x = x[:-1] + u/2
pos_y = y[:-1] + v/2
norm = np.sqrt(u**2+v**2) 

fig, ax = plt.subplots(); ax.plot(x,y, marker="o"); ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid"); ax.set_xlim([x_min,x_max]); ax.set_ylim([y_min,y_max])
plt.savefig('Time_Steps_9-20.png',dpi=200)

#ARROWS
x = df_b.S_x.iloc[19:32].values
y = df_b.S_y.iloc[19:32].values

u = np.diff(x)
v = np.diff(y)
pos_x = x[:-1] + u/2
pos_y = y[:-1] + v/2
norm = np.sqrt(u**2+v**2) 

fig, ax = plt.subplots(); ax.plot(x,y, marker="o"); ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid"); ax.set_xlim([x_min,x_max]); ax.set_ylim([y_min,y_max])
plt.savefig('Time_Steps_19-32.png',dpi=200)

#ARROWS
x = df_b.S_x.iloc[31:45].values
y = df_b.S_y.iloc[31:45].values

u = np.diff(x)
v = np.diff(y)
pos_x = x[:-1] + u/2
pos_y = y[:-1] + v/2
norm = np.sqrt(u**2+v**2) 

fig, ax = plt.subplots(); ax.plot(x,y, marker="o"); ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid"); ax.set_xlim([x_min,x_max]); ax.set_ylim([y_min,y_max])
plt.savefig('Time_Steps_31-45.png',dpi=200)


'''ALL TOGETHER'''
#ARROWS
x = df_b.S_x.iloc[2:].values
y = df_b.S_y.iloc[2:].values

u = np.diff(x)
v = np.diff(y)
pos_x = x[:-1] + u/2
pos_y = y[:-1] + v/2
norm = np.sqrt(u**2+v**2) 

fig, ax = plt.subplots(); ax.plot(x,y, marker="o"); ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid"); ax.set_xlim([x_min,x_max]); ax.set_ylim([y_min,y_max])
plt.savefig('Time_Steps_All.png',dpi=200)





'''DEV'''
'''
ISSUE: JUSTIFYINIG POLYNOMIAL OF HIGHER DEG
0. deg of poly does not change epi transformation
1. Test if lower deg poly on human will still give same relation to epii IT DOESNT
2. If yes, use it.
3. If no, then can justify using higher bc it fits epi, but then 
must show not spurious--> test random field (using var of hum) and fit same set 
of higher degree poly and match to epi.  If not get same fit, then we are ok. 
Will use Monte Carlo with 1000 runs to give probability that we can randomly get
a harmonic oscilllator.
X. May use family  of poly for match to epi
X. May use more advanced regression methods with penalization or regularization.
X. May check sensitivity of poly to values
CONVINCING WAY TO ADJUDICATE THE USE OF HIGH DEG POLYNOMIALL


'''

#deg_list = [10,11,12,13,14,15,16,17]
#deg_list = [10,11,12,13]
deg_list = [2,3,4,5,6,7]
for i in deg_list:
    x_poly = np.arange(len(hum_use))+1
    hum_poly = np.polyfit(x_poly, hum_use, deg=i,full=True)#deg 15 worked
    hum_poly_values = np.polyval(hum_poly[0], x_poly)
    plt.plot(hum_poly_values,label=i)
    print('RESIDUALS SUM: ',hum_poly[1])
    plt.legend()
plt.scatter(np.arange(len(hum_use)),np.array(hum_use))
plt.savefig('poly_degree_human_tmp.png',dpi=150)

plt.plot(np.array(hum_use))
plt.plot(np.array(grouped.hum.mean().rolling(4).mean()),color='red',label='prior',dashes=(0,2,2,2))


for i in deg_list:
    x_poly = np.arange(len(vdh_use))+1
    vdh_poly = np.polyfit(x_poly, vdh_use, deg=i, full=True)
    vdh_poly_values = np.polyval(vdh_poly[0], x_poly)
    plt.plot(vdh_poly_values,label=i)
    plt.legend()
plt.scatter(np.arange(len(vdh_use)),np.array(vdh_use))


hum_poly = np.polyfit(x_poly, hum_use, deg=15,full=True)
hum_poly[1]

plt.plot(hum_use)
plt.scatter(hum_use.index[1:],hum_use[1:])







'''OLD AND SCRAP CODE'''

'''OLD AND SCRAP CODE'''

'''OLD AND SCRAP CODE'''
#OLD CODE
x_poly = np.arange(len(hum_use))+1
hum_poly = np.polyfit(x_poly, hum_use, deg=15)#deg 15 worked
plt.plot(hum_poly)
hum_poly_values = np.polyval(hum_poly, x_poly)
plt.plot(hum_poly_values)
plt.plot(hum_use)

h_x1 = np.diff(hum_poly_values)
h_x1_S = pd.Series(h_x1,index=hum_use.index[1:])

h_x2 = np.diff(h_x1)
h_x2 = np.insert(h_x2,0,0,axis=0)
plt.plot(h_x1); plt.plot(h_x2);  plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)

h_x2_S = pd.Series(h_x2,index=hum_use.index[2:])
plt.plot(h_x2_S); plt.plot(hum_use[2:]); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)

plt.plot(h_x2_S);  plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)
plt.savefig('h_x2_S.png', dpi=300, transparent=False, bbox_inches='tight')

plt.plot(v_x2_S);  plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)
plt.savefig('v_x2_S.png', dpi=300, transparent=False, bbox_inches='tight')



'''
EXPLORATION
'''
norm_const = 0.2

v_const = 0.01; h_const = 5; plt.plot(v_x1*v_const); plt.plot(h_x1*h_const); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)

v_const = 0.05; h_const = 5; plt.plot(v_x2*v_const); plt.plot(h_x2*h_const); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)

'''*****BINGO BINGO BINGO'''
#EPI VEL, HUMAN VEL
v_const = 0.05; h_const = 5; plt.plot(v_x1*v_const,label='Epi Vel.'); plt.plot(h_x1*h_const, label='Hum Vel.'); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)
plt.legend()

#EPI VEL, HUMAN Max-Hum Vel
v_const = 0.05; h_const = 5; plt.plot(v_x2*v_const,label='Epi Acc.'); plt.plot(h_x1*h_const, label='Hum Vel.'); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)
plt.legend()


#EPI ACC, HUMAN ACC
v_const = 0.05; h_const = 5; plt.plot(v_x2*v_const,label='Epi Acc.'); plt.plot(h_x2*h_const, label='Hum Vel.'); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)
plt.legend()

#EPI ACC, HUMAN VEL
v_const = 0.05; h_const = 5; plt.plot(v_x2*v_const,label='Epi Acc.'); plt.plot(h_x1*h_const, label='Hum Vel.'); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)
plt.legend()
plt.savefig('Epi_Acc_w_Hum_Vel.png',dpi=200)

#EPI VEL, HUMAN ACC
v_const = 0.05; h_const = 50; plt.plot(v_x1*v_const,label='Epi Vel.'); plt.plot(h_x2*h_const, label='Hum Acc.'); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)
plt.legend()
plt.savefig('Epi_Vel_w_Hum_Acc.png',dpi=200)

#SCATTERS AND THINGS
df_b = pd.DataFrame([v_x1_S,h_x2_S]).T
plt.scatter(df_b[0],df_b[1])

#CORRELATION DOESN"T WORK BC DYNAMIC IS LOST
stats.pearsonr(v_x1, h_x2)

#EOF
#EFO
#EOF