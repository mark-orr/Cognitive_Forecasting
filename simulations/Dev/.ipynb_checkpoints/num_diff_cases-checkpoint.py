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

cases_data_in = '/Users/biocomplexity/Projects/SocioCognitiveModeling/Metaculus_CogModeling/simulations/InputData'

'''EPI DATA'''
vdh = pd.read_csv(f'{cases_data_in}/VDH-COVID-19-PublicUseDataset-Cases.csv',parse_dates=['report_date'])

vdh = vdh.groupby(['report_date'])['total_cases'].sum().diff().rolling(window=7).mean().round()
vdh_use = vdh['11-30-2021':'01-14-2022'].copy()
vdh_use.name = 'use'

#PROCESS RAW DATA FOR DIFFERENTIATION
x_poly = np.arange(len(vdh_use))+1
vdh_poly = np.polyfit(x_poly, vdh_use, deg=15)
vdh_poly_values = np.polyval(vdh_poly, x_poly)

#FIRST DERIVATIVE
v_x1 = np.diff(vdh_poly_values)
v_x1_S = pd.Series(v_x1,index=vdh_use.index[1:])

#SECOND DERIVATIVE
v_x2 = np.diff(v_x1)
v_x2 = np.insert(v_x2,0,0,axis=0)#NOW SAME LEN AS v_x1 and indexes matched in time.
v_x2_S = pd.Series(v_x2,index=vdh_use.index[1:])

#SANITY PLOTS
plt.plot(v_x1_S); plt.plot(v_x2_S); plt.plot(vdh_use[2:]*0.1); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)
plt.plot(v_x1_S); plt.plot(vdh_use[2:]*0.1); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)
plt.plot(v_x2_S);  plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)


'''ADD THE HUMAN DATA'''
'''ADD THE HUMAN DATA'''
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


'''
HUMAN DATA 
FOR DIFFERENTIIATION
'''

#TRY NON SMOOTHED DATA
hum_use = grouped.hum.mean()

#PROCESS RAW DATA FOR DIFFERENTIATION
x_poly = np.arange(len(hum_use))+1
hum_poly = np.polyfit(x_poly, hum_use, deg=15)
hum_poly_values = np.polyval(hum_poly, x_poly)

#FIRST DERIVATIVE
h_x1 = np.diff(hum_poly_values)
h_x1_S = pd.Series(h_x1,index=hum_use.index[1:])

#SECOND DERIVATIVE
h_x2 = np.diff(h_x1)
h_x2 = np.insert(h_x2,0,0,axis=0)#NOW SAME LEN AS v_x1 and indexes matched in time.
h_x2_S = pd.Series(h_x2,index=hum_use.index[1:])

#FOR RAPID TESTING AGAINST EPI DATA
v_const = 0.05; h_const = 20; plt.plot(v_x1*v_const,label='Epi Vel.'); plt.plot(h_x2*h_const, label='Hum Acc.'); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)

#SANITY PLOTS
plt.plot(h_x1_S); plt.plot(h_x2_S); plt.plot(hum_use[2:]*1); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)
plt.plot(h_x1_S); plt.plot(hum_use[2:]*0.1); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)
plt.plot(h_x1); plt.plot(h_x2); plt.plot(hum_poly_values); plt.plot(np.array(hum_use[1:]))
plt.plot(h_x2_S);  plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)

'''HUMAN ROLLING SMOOTH IS NO GOOD'''
#A TEST OF HUMAN SMOOTHING
tmp_x1 = np.diff(np.array(grouped.hum.mean().rolling(4).mean()))
tmp_x2 = np.diff(tmp_x1)
tmp_x2 = np.insert(tmp_x2,0,0,axis=0)
tmp_x2_S = pd.Series(tmp_x2,index=hum_use.index[1:])
#FOR RAPID TESTING AGAINST EPI DATA
v_const = 0.05; h_const = 20; plt.plot(v_x1*v_const,label='Epi Vel.'); plt.plot(tmp_x2*h_const, label='Hum Acc.'); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)
plt.plot(tmp_x1); plt.plot(np.array(grouped.hum.mean().rolling(4).mean()[1:]))
plt.plot(tmp_x1)



'''
PLOTTING EPI AND 
HUMAN TOGETHER
'''

norm_const = 0.2

v_const = 0.01; h_const = 5; plt.plot(v_x1*v_const); plt.plot(h_x1*h_const); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)

v_const = 0.05; h_const = 5; plt.plot(v_x2*v_const); plt.plot(h_x2*h_const); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)

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

#ROLLING CORRELATION ON PERCENT CHANGE
v_x1_pc = v_x1_S.pct_change()
h_x2_pc = h_x2_S.pct_change()
v_h_corr = v_x1_pc.rolling(4).corr(h_x2_pc)
plt.plot(v_h_corr)


'''GOOD PERCENT CHANGE THINGIE'''
h_u_pc = hum_use.pct_change()
v_u_pc = vdh_use.pct_change()
for i in range(3,11):
    v_u_h_u_corr = v_u_pc.rolling(i).corr(h_u_pc)
    plt.plot(v_u_h_u_corr,label=i)
    plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=1)
    plt.legend()
    
plt.plot(np.array(h_u_pc),label='hum'); plt.plot(np.array(v_u_pc),label='epi'); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2); plt.legend()
plt.scatter(np.arange(len(h_u_pc)),np.array(h_u_pc)); plt.scatter(np.arange(len(v_u_pc)),np.array(v_u_pc)); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2); plt.legend()
'''TRY COMET TAIL'''
y = np.array(h_u_pc)
x = np.array(v_u_pc)
alpha_correction = 1/len(x)
for i in range(0,len(x)): plt.scatter(x[i],y[i],alpha=alpha_correction*i,color='black',marker='+'); plt.plot(x,y,linewidth=0.2,alpha=0.5)
plt.savefig('tmp.png',dpi=300)









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
d1 = pd.Series(np.array(hum_use.pct_change()))
d2 = pd.Series(np.array(vdh_use.pct_change()))



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
d1 = hum_use.interpolate().values
d2 = vdh_use.interpolate().values
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


#NO ARROWS
x = df_b.S_x
y = df_b.S_y

fig, ax = plt.subplots()
ax.scatter(x,y)
ax.plot(x,y)
alpha_correction = 1/len(x)
'''TRY COMET TAIL'''
for i in range(0,len(x)): plt.scatter(x[i],y[i],alpha=alpha_correction*i,color='black',marker='+'); plt.plot(x,y,linewidth=0.2,alpha=0.5)
plt.savefig('tmp_phase_space.png',dpi=300)

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
deg_list = [10,14,15,16]
for i in deg_list:
    x_poly = np.arange(len(hum_use))+1
    hum_poly = np.polyfit(x_poly, hum_use, deg=i,full=True)#deg 15 worked
    hum_poly_values = np.polyval(hum_poly[0], x_poly)
    plt.plot(hum_poly_values)
    print('RESIDUALS SUM: ',hum_poly[1])
    
plt.scatter(np.arange(len(hum_use)),np.array(hum_use))
plt.plot(np.array(hum_use))
plt.plot(np.array(grouped.hum.mean().rolling(4).mean()),color='red',label='prior',dashes=(0,2,2,2))


for i in deg_list:
    x_poly = np.arange(len(vdh_use))+1
    vdh_poly = np.polyfit(x_poly, vdh_use, deg=15)
    vdh_poly_values = np.polyval(vdh_poly, x_poly)
    plt.plot(vdh_poly_values)


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


#EOF
#EFO
#EOF