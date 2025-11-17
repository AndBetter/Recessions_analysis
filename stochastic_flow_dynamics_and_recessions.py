# This code imports a single precipitation and streamflow time series for 
# catchments in either the CAMEL or MOPEX datasets. It computes the frequency 
# and intensity of runoff events (streamflow increments) as well as the 
# recession properties of the hydrograph. Recessions are modeled as a power 
# law with the equation dq/dt = a*q^b. With some basic modifications, 
# this code can be utilized to quantify changes in the frequency/intensity of
# runoff events and the recession properties of catchments - for example - before and after
#  a wildfire.

# works with the SPATIAL environment 


#import packages
import numpy as np
import pandas as pd
from pathlib import Path
import glob, os
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import re
from scipy.ndimage import label
from scipy.optimize import curve_fit
import datetime


def extract_recessions(q, min_length=5, min_q_start=0):
    '''
    extract recesions from an hydrograph
    min lenght of the recession and minimum discharge at the beginning of the 
    recession can be set
    '''
  
    #forward difference
    q_diff = np.diff(q,append=np.nan)
    #decreasing part of the hydrograph
    q_decreasing = q_diff<0
    # Label connected components to identify recessions
    labeled_array, num_features = label(q_decreasing)
    
    # Initialize result array
    q_decreasing_filtered = q_decreasing.copy()

    # Iterate over each labeled component and remove those not matching with the selection criteria
    for component_label in range(1, num_features + 1):
        # Get indices of the current component
        indices = np.where(labeled_array == component_label)[0]
        
        # Check conditions on duration and Q0
        if len(indices) < min_length or np.max(q[indices])<min_q_start :
            # Replace short sequence with 0s
            q_decreasing_filtered[indices] = 0

    # Label connected components on the filtered sequence
    labeled_array_filtered, num_features_filtered = label(q_decreasing_filtered)

    recessions=[]
    for component_label in range(1, num_features_filtered + 1):
        indices = np.where(labeled_array_filtered == component_label)[0]
        recessions.append(q[indices])


    return recessions  








#Define seasons:
seasons = {
    "spring": [3,4,5],
    "summer": [6,7,8],
    "autumn": [9,10,11],
    "winter": [12,1,2],
    "year"  : [1,2,3,4,5,6,7,8,9,10,11,12]
          }


#<<<<<< denotes where action is needed 

#select dataset  <<<<<<<
#MOPEX: https://www.hydroshare.org/resource/99d5c1a238134ea6b8b767a65f440cb7/
#CAMEL: https://gdex.ucar.edu/dataset/camels.html

dataset = 'MOPEX'  # options: MOPEX, CAMEL 


#select usgs gauge  <<<<<<<
gauge_name = '3512000'    # MOPEX
gauge_name = '3455000'    # MOPEX
#gauge_name = '13351000'   # MOPEX
#gauge_name = '12413000'   # MOPEX
#gauge_name = '9497500'    # MOPEX
#gauge_name = '1611500'    # MOPEX
#gauge_name = '3465500'   # MOPEX
#gauge_name = '3512000'  # MOPEX
#gauge_name = '8189500'  # MOPEX
#gauge_name = '9430500'  # MOPEX
#gauge_name = '7163000'  # MOPEX
gauge_name = '02143040'  # MOPEX
gauge_name = '11160000'  # MOPEX
#gauge_name = '01445000'  # MOPEX
#gauge_name = '03504000' # MOPEX
#gauge_name = '02138500' # MOPEX
#gauge_name = '02143040' #MOPEX
gauge_name = '5526000'  # MOPEX  
#gauge_name = '********' 

#choose a season <<<<<<<
season     = 'year'


#import data depending on the dataset


if dataset == 'CAMEL':
    
    #set dataset path <<<<<<
    dataset_folder = Path('E:/Padova/datasets/CAMEL')
    
    
    ###########################
    ### IMPORT STREAMFLOW DATA
    ###########################
       
    #extract streamflow path for selected gauge
    streamflow_file =  glob.glob( str( dataset_folder / 'usgs_streamflow' / '**' / f'*{gauge_name}*.txt' ) , recursive =False)[0]  
    
    
    # Define columns names
    column_names_q = ["station", "year", "month", "day", "flow", "?"]
    
    
    # Read the text file
    streamflow = pd.read_csv( streamflow_file , sep=r"\s+", names=column_names_q) 
    
    
    #extract flow data for the specific season
    seasonal_flow  =  streamflow[streamflow.month.isin(seasons[season])]
    
    #create a datetime object from database dates
    flow_dates   =   [ datetime.datetime(seasonal_flow.year.iloc[i], seasonal_flow.month.iloc[i], seasonal_flow.day.iloc[i])   for i in range(len(seasonal_flow)) ]
    
    
    
    #####################
    ### IMPORT METEO DATA
    #####################
    
    
    # options for climatic forcings, see dataset readme : daymet, maurer, nldas  <<<<<<<<
    
    
    #extract streamflow path of selected gauge
    meteo_file =  glob.glob( str( dataset_folder / 'basin_mean_forcing' / 'daymet' / '**' / f'*{gauge_name}*.txt' ) , recursive =False)[0]  
    
    # Define column names
    column_names_m = ["year", "month", "day", "hour",  "?", "h", 'srad', 'swe', 'tmax', 'tmin', 'vp']
    
    # Read the text file
    meteo = pd.read_csv(meteo_file, skiprows=4, sep=r"\s+", names=column_names_m)
    
    #extract flow data for the specific season
    seasonal_meteo  =  meteo[meteo.month.isin(seasons[season])]
    
    #create a datetime object from database dates
    meteo_dates   =   [ datetime.datetime(seasonal_meteo.year.iloc[i], seasonal_meteo.month.iloc[i], seasonal_meteo.day.iloc[i])   for i in range(len(seasonal_meteo)) ]


    
    
    # keeps just the part of the flow and meteo timeseries that are syncrhonous
    
    meteo_dates_min = min(meteo_dates)
    meteo_dates_max = max(meteo_dates)
    
    flow_dates_min =  min(flow_dates)
    flow_dates_max =  max(flow_dates)
    
    timeseries_min =  max(meteo_dates_min, flow_dates_min )
    timeseries_max =  min(meteo_dates_max, flow_dates_max )
    
    
    seasonal_meteo = seasonal_meteo.iloc[ (np.array(meteo_dates)> timeseries_min) & (np.array(meteo_dates) < timeseries_max)]
    seasonal_flow  = seasonal_flow.iloc[ (np.array(flow_dates)> timeseries_min) & (np.array(flow_dates) < timeseries_max)]
    
    #identifies where there are synchronous precipitation and discharge data available
    where_data =  (seasonal_flow.flow >=0) & (seasonal_meteo.h >=0)
    
    #........
   
    #extract precipitation 
    h = np.array(seasonal_meteo.h)
    
    
    #extract catchment area in skm
    with open(meteo_file, 'r') as file:
        catchment_area = file.readlines()
        catchment_area = catchment_area[2]
        catchment_area = int(str(re.findall(r'\d+', catchment_area)[0] )  ) / 1000**2
    
    
    
    #goes from usgs flow to specific discharge (i.e. mm/day)
    q = np.array(seasonal_flow.flow)* 0.3048**3   *3600 * 24 / (catchment_area* 1000**2) * 1000
    
    




elif dataset == 'MOPEX':
      
    
    #set dataset path <<<<<<<
    dataset_folder = Path('E:/Padova/datasets/MOPEX')
    

    ####################################
    ### IMPORT STREAMFLOW and METEO DATA
    ####################################
    
    #extract streamflow path for selected gauge
    streamflow_file =  glob.glob( str( dataset_folder / 'MOPEX' / 'Daily' / f'*{gauge_name}*.dly' ) , recursive =False)[0]  
    
    
    # Define columns names
    column_names_q = [ "year", "month", "day", "precipitation", "PET", "flow", "T_max", "T_min"]
    
    
    # Read the text file
    colspecs=[(0, 4), (4, 6), (6, 8), (8,18), (18,28), (28,38), (38,48), (48,58)]
    streamflow = pd.read_fwf( streamflow_file , colspecs=colspecs , names=column_names_q) 
    
    
    #extract flow data for the specific season - extract just where data for both discharge and precipitation are available
    seasonal_flow  =  streamflow[(streamflow.month.isin(seasons[season])) & (streamflow.flow >=0) & (streamflow.precipitation >=0) ]
    
    
    
    
    #create a datetime object from database dates
    flow_dates   =   [ datetime.datetime(seasonal_flow.year.iloc[i], seasonal_flow.month.iloc[i], seasonal_flow.day.iloc[i])   for i in range(len(seasonal_flow)) ]
    
    
    q   = np.array(seasonal_flow.flow)
    h   = np.array(seasonal_flow.precipitation)
    PET = np.array(seasonal_flow.PET)


    ###############################
    ### IMPORT CATCHMENT ATTRIBUTES
    ###############################
    catchment_attributes = pd.read_csv(dataset_folder / 'elevation_slope_mopex431.txt', sep='\t')


    print(catchment_attributes[catchment_attributes.USGS_SiteCode== int(gauge_name)])

################## end import ##################################################
################################################################################



#neglects precipitation below 1mm/day  
h[h<1]=0


#identifies when precipitation occurs and extract precipitation depth
h_peak_indices = np.where(h>0)[0] 
h_peak         = h[h_peak_indices]  



##find peaks -- identifies positive increment  in flow timeseries --  method 1

# Compute forward difference
q_diff = np.diff(q, n=1)
q_diff = np.pad(q_diff, (1, 0), 'constant', constant_values=(0, 0))

q_diff_forward = np.diff(q, n=1) #np.diff(q[::-1], n=1)[::-1]
q_diff_forward = np.pad(q_diff_forward, (0, 1), 'constant', constant_values=(0, 0))# np.pad(q_diff_forward, (1, 0), 'constant', constant_values=(0, 0))

#positive increments below this value are neglected <<<<<<<<
q_diff_positive_min  = np.percentile ( q_diff[q_diff>0],95) / 100
#q_diff_positive_min  = 0.001


#q_peak_indices   = np.where(q_diff>q_diff_positive_min  )[0] 
q_peak_indices   = np.where((q_diff>q_diff_positive_min)  & (  (h>0) | (np.roll(h, 1) >0  ) )  )[0] 
#q_peak_indices   = np.where((q_diff>q_diff_positive_min)  & (  (h>0) )  )[0] 
q_diff_positive  = q_diff[q_peak_indices]
q_peak           = q[q_peak_indices]  




# =============================================================================
# ##find peaks --  method 2  DA PERFEZIONARE
# 
# local_maxima   = (q[1:-1] >= q[:-2]) & (q[1:-1] >= q[2:])
# local_maxima   = np.pad(local_maxima, pad_width=(1, 1), mode='constant', constant_values=0)
# local_maxima   = np.where(local_maxima)[0]
# q_local_maxima = q[local_maxima]
# 
# local_minima   = (q[1:-1] <= q[:-2]) & (q[1:-1] <= q[2:])
# local_minima   = np.pad(local_minima, pad_width=(1, 1), mode='constant', constant_values=0)
# local_minima   = np.where(local_minima)[0]
# q_local_minima = q[local_minima]
# 
# delta_q= np.zeros(len(q))
# for lm, qlm in zip(local_maxima, q_local_maxima) :
#     diff_time = lm - local_minima
#     if (diff_time[diff_time>0] >0).size:
#         di_time = np.min( diff_time[diff_time>0] )
#         delta_q[lm] = q[lm]-q[lm-di_time]
# =============================================================================



######
# Plot
######

fig, ax1 = plt.subplots()

# Plot the first series on the left axis
ax1.plot(q)
ax1.scatter(q_peak_indices,q_peak, s=5, c='r')
ax1.bar(q_peak_indices,q_diff_positive, alpha=0.5, color='blue')
ax1.set_ylabel('q (mm/day)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.axis(ymin=0,ymax=np.percentile(q,99))

# Create a second axis on the right side
ax2 = ax1.twinx()

# Plot the second series on the right axis
ax2.bar(h_peak_indices, h_peak, alpha=0.5, color='red')
ax2.set_ylabel('h (mm/day)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Set the bottom axis as the top-down axis
ax2.invert_yaxis()

# Label x-axis and title
plt.title('...')
plt.xlabel('Time')
plt.xlim(1000, 1200)

# Display the plot
plt.show()


###############################################################################
###############################################################################

# compute the parameter of the stochastic flow dynamics model

#average freqency (lambda) and intensity (alpha) of streamflow increments
lambda_Q_t = len(q_diff_positive>0)/len(q)  # (1/(day))
alpha_Q_t  = np.mean(q_diff_positive)       # (mm)     !!!!!!!! questa non è la definizione di alpha_Q_t !!!!!!


#average freqency (lambda) and intensity (alpha) of precipitation
lambda_P_t = len(h_peak)/len(h)
alpha_P_t  = np.mean(h_peak)




#extracts recessions (i.e. when flow is decreasing and there is no precipitation)
q_recession = np.copy(q)*0
#q_recession[(q_diff<0) & (h == 0)   ] = q[(q_diff<0) & (h == 0)  ]
q_recession[q_diff_forward<0 ] = q[q_diff_forward<0 ]
#q_recession[(q_diff_forward<0) & (h == 0)   ] = q[(q_diff_forward<0) & (h == 0)  ]
#q_recession[(q_diff<0) & (h == 0) & (q< np.percentile(q,95)  )  ] = q[(q_diff<0) & (h == 0) & (q< np.percentile(q,95) ) ]


# Labels each recession
labels, num_components = label(q_recession)


# organizes the recessions in a 2d array and in a single 1d array
t_max = 100
q_recessions = np.zeros((num_components,t_max))
q_before_recessions = np.zeros((num_components))
q_before_recessions_location = np.zeros((num_components)).astype(int)
q_recession_all = np.zeros((num_components*t_max,2))

recession_all=[]



for i in range(1, num_components + 1):
    temp = q[np.where(labels == i)[0]]
    q_recessions[i-1,0:len(temp)]=  temp
    q_before_recessions_location[i-1]  = np.min(np.where(labels == i))-1
    q_before_recessions[i-1] = q[q_before_recessions_location[i-1]]
    
# excludes recession shorther than min_recession_length days <<<<<<<
min_recession_length = 5 

row_to_pop = np.sum(q_recessions>0,axis=1)<min_recession_length

# Create a new array without the selected row
q_recessions = np.delete(q_recessions, row_to_pop, axis=0)
q_recessions[q_recessions==0]=np.nan
q_before_recessions = np.delete(q_before_recessions, row_to_pop)
q_before_recessions_location = np.delete(q_before_recessions_location, row_to_pop)

q_recessions_diff = -np.diff(q_recessions, n=1)

q_recessions=q_recessions[:,0:-1]
q_recessions[np.isnan(q_recessions_diff)]=np.nan




#####################
## PLOT recessions #1
#####################
for i in range(len(q_recessions)):
    plt.plot(np.array(range(t_max-1)), q_recessions[i], label='Line {}'.format(i+1))

# Add labels and legend
plt.title('all recessions')
plt.xlabel('time since inception (day)')
plt.ylabel('q (mm/day)')
plt.show()


q_recessions_all      = q_recessions.flatten()[~np.isnan(q_recessions.flatten())]
q_recessions_diff_all = q_recessions_diff.flatten()[~np.isnan(q_recessions_diff.flatten())]




##############################
##  FIT ALL RECESSIONS AT ONCE 
##############################

# option 1: exponential recessions
def func_exp(x, k):
    return  k*x
[k_cloud], pcov = curve_fit(func_exp,  q_recessions_all ,  q_recessions_diff_all , maxfev=100000  ) 


# =============================================================================
# # option 1bis: exponential recessions with intercept
# def func_exp_intercept(x,  k0,k):
#     return k0+ k*x
# 
# [k0,k], pcov = curve_fit(func_exp_intercept,  q_recessions_all ,  q_recessions_diff_all , maxfev=100000  ) 
# 
# =============================================================================

# =============================================================================
# # option 1tris:  exponential regression on logtransformed data--  fits the logtransform data (more weight to low flows) and force the intecept to be 0
# def func_exp(x, k):
#     return k  + x
# 
# [k], pcov = curve_fit(func_exp,  np.log(q_recessions_all) ,  np.log(q_recessions_diff_all) , maxfev=10000  ) 
# k_cloud=np.exp(k)
# =============================================================================



# option 2: power law regression 
def func_pow(x,  a,b):
    return a*x**b

[a_cloud,b_cloud], pcov = curve_fit(func_pow, ( q_recessions_all) , ( q_recessions_diff_all) , maxfev=10000  ) 




# =============================================================================
# # option 2bis:  power law regression on logtransformed data--  fits the logtransform data (more weight to low flows) and force the intecept to be 0
# def func_pow_log(x,  a,b):
#     return a+ b*x
# 
# [a,b], pcov = curve_fit(func_pow_log,  np.log(q_recessions_all) , np.log( q_recessions_diff_all)  ) 
# 
# a = np.exp(a)
# 
# =============================================================================



###########################################################################
###########################################################################




##################################
##  ALTERNATIVE, FIT EACH RECESSION
##################################


K= np.zeros(q_recessions.shape[0])
A= np.zeros(q_recessions.shape[0])
B= np.zeros(q_recessions.shape[0])
recession_duration = np.zeros(q_recessions.shape[0])
q_mean_recession   = np.zeros(q_recessions.shape[0])


for i in range(len(K)):
    temp_1 = q_recessions[i,:][q_recessions[i,:]>0]
    temp_2 = q_recessions_diff[i,:][q_recessions_diff[i,:]>0]
    [K[i]], pcov= curve_fit(func_exp,  temp_1 ,  temp_2 , maxfev=100000 )
    [A[i], B[i]], pcov = curve_fit(func_pow,  temp_1 ,  temp_2  , maxfev=100000 ) 
   
    #[A[i], B[i]], pcov = curve_fit(func_pow_log,  np.log(temp_1) ,  np.log(temp_2)  ) 
    #A[i] =np.exp(A[i])
    
    q_mean_recession[i]   = np.max(temp_1)
    recession_duration[i] = np.shape(temp_1)[0]
    

    
    plt.scatter( temp_1, temp_2)
    plt.ylim(np.percentile(q_recessions_diff_all,0), np.percentile(q_recessions_diff_all,100))
    plt.xlim(np.percentile(q_recessions_all,0), np.percentile(q_recessions_all,100))
    plt.xscale('log')  # Set the x-axis to use a logarithmic scale
    plt.yscale('log')  # Set the y-axis to use a logarithmic scale
    #plt.plot( temp_1, K[i] * temp_1 ,color='red' )
    #plt.plot( temp_1, A[i] * temp_1**B[i], color='blue' )
    #plt.show()  # UNCOMMENT THIS TO PLOT ONE BY ONE 

plt.show()
    
k= np.median(K)
a= np.median(A)
b= np.median(B)
 




#fit power low params
def fit_params(x,  aa,yy):
    return yy + aa/(x)
initial_guess = [2, 2]

[aa,yy], pcov = curve_fit(fit_params,  q_mean_recession[B>0] , B[B>0], maxfev=100000, p0=initial_guess ) 



##PLOT
x = np.arange(0.5,q_mean_recession.max(),0.1)
plt.plot(x,yy + aa/(x))
plt.scatter(q_mean_recession[B>0.00], B[B>0.00], marker ='.' , c=(q_before_recessions[B>0.00]/q_mean_recession[B>0.00]), cmap='viridis')
#plt.scatter(q_before_recessions[B>0.00], B[B>0.00], marker ='.' , c=np.log(q_mean_recession[B>0.00]))
#plt.scatter(q_mean_recession, K, marker ='+')
plt.ylabel('b;a')
plt.xlabel('q_max ?')
#plt.xscale('log')  
#plt.yscale('log')  
plt.show()




###########################################################################
###########################################################################






# Generate the trend lines
temp = np.arange(q_recessions_all.min(),q_recessions_all.max(),0.001)

line_exp   = k_cloud * temp  
#line_exp_intercept = k0 + k*temp
line_power = a * temp**b 
line_power_cloud = a_cloud * temp**b_cloud 





##PLOT
plt.scatter(q_recessions_all, q_recessions_diff_all, marker ='.')
plt.plot(temp, line_exp, color='red', label='exp fit cloud')
#plt.plot(temp, line_exp_intercept, color='green', label='Fitted Line_exp_intercept')
plt.plot(temp, line_power, color='blue', label='power fit single')
plt.plot(temp, line_power_cloud, color='green', label='power fit cloud')
# Add labels and legend
plt.xlabel('q (mm/day)')
plt.ylabel('dq/day')
plt.ylim(np.percentile(q_recessions_diff_all,0), np.percentile(q_recessions_diff_all,100))
plt.xlim(np.percentile(q_recessions_all,0), np.percentile(q_recessions_all,100))
plt.legend(loc='lower right')
plt.xscale('log')  # Set the x-axis to use a logarithmic scale
plt.yscale('log')  # Set the y-axis to use a logarithmic scale
plt.show()


################################################################################
##############################################################################
## GENERATE STOCHASTIC FLOW DYNAMICS
#  this part uses the parameters lambda, alpha and k estimated from the previous
#  observed streamflow timeseries  and generates a stochastic flow timeseries with 
#  the same average frequency  and intensity of flow pulses as in the observed record
##############################################################################
###############################################################################


#  how long you want you synthetic timeseries to be 
timeseries_length  = 10000    
k = k_cloud
a = a
b = b

# generate expoentially distributed interarrivals between runoff events with a 1-day discretization
def discrete_exponential(lam, size):
    u = np.random.rand(size)  # Generate uniform random numbers between 0 and 1
    x = np.floor(-np.log(1 - u) / lam)  # Discretize the exponential distribution
    return x.astype(int)




interarrivals = discrete_exponential(lambda_Q_t,int(timeseries_length*(lambda_Q_t+0.1)))





# generate expoentially distributed interarrivals between runoff events
#interarrivals = np.round(  np.random.exponential(  1/lambda_Q_t, size= int(timeseries_length*(lambda_Q_t+0.1))  )  ).astype(int)





# generates expoentially distributed runoff spikes
spikes =   np.random.exponential(  alpha_Q_t, size= int(timeseries_length*(lambda_Q_t+0.1))  )


#one timeseries with power-law recessions and one with expoenential recessions
q_simulated_exp     = np.zeros((timeseries_length))
q_simulated_power   = np.zeros((timeseries_length))
t_spikes            = np.cumsum( interarrivals )



for t in range(1,timeseries_length,1):
    
    sum_spikes = np.sum(spikes[t_spikes == t])
    
    #############################
    ## exponential recessions
    ##############################
    #q_simulated_exp[t]   = q_simulated_exp[t-1] - q_simulated_exp[t-1] * (1-np.exp(-k*0.5))  + sum_spikes  - (q_simulated_exp[t-1] - q_simulated_exp[t-1] * (1-np.exp(-k*0.5))  + sum_spikes) * (1-np.exp(-k*0.5))
    #q_simulated_exp[t]   = q_simulated_exp[t-1]  + sum_spikes  -   q_simulated_exp[t-1] * (1-np.exp(-k*1))
    q_simulated_exp[t] = (q_simulated_exp[t-1]*(np.exp(-k*0.5)) + sum_spikes ) /k * (1- np.exp(-k*1))

    ######################
    ##power-law recessions
    ######################
    q_simulated_power[t] = np.nanmax( [q_simulated_power[t-1] - a*q_simulated_power[t-1]**b *1   ,1e-30]) + sum_spikes
    
    #q_simulated_power[t] = np.nanmax( [q_simulated_power[t-1] - a*q_simulated_power[t-1]**b *0.5 + sum_spikes - a*(q_simulated_power[t-1] - a*q_simulated_power[t-1]**b *0.5 + sum_spikes)**b *0.5  ,1e-30])
    

    #c   = np.max( [ q_simulated_power[t-1]  + sum_spikes  -  a*q_simulated_power[t-1]**(b)  , 0 ] )    
    #q_simulated_power[t] = ((np.max ( [q_simulated_power[t-1] +sum_spikes,  1e-30]))**(1-b)  -  a*(1-b)*1 )**(1/(1-b))
    
    ##this works ok
    #q_simulated_power[t] = np.nanmax([( np.nanmax( [ q_simulated_power[t-1],1e-30])**(1-b) + a*(b-1)*1   ),1e-30]) **(1/(1-b)) + sum_spikes
   
    
## PLOT TIMESERIES
plt.plot(q_simulated_exp, color='red',label='q_simulated_exp')
plt.plot(q_simulated_power, color='blue',label='q_simulated_power')
plt.plot(q, color='black',label='q_observed')
plt.ylim(0, np.percentile(q,99))
plt.xlim(2000, 2300)
plt.xlabel('time (days)')
plt.ylabel('q (mm/day)')
plt.legend(loc='upper right')
plt.show()


q_mean = np.mean(q[q>0])
q_simulated_exp_mean = np.mean(q_simulated_exp)
q_simulated_pow_mean = np.mean(q_simulated_power)




## PLOT PDF
#define bins of the histogram
temp1 = 0
temp2 = np.percentile(q,99)
num_bins = 30
step = (temp2 - temp1) / num_bins


logg = False
plt.hist(q_simulated_exp,   bins=np.arange(temp1,temp2,step), color='red',  density=True, alpha = 0.5,label='q_simulated_exp', log=logg)
plt.hist(q_simulated_power, bins=np.arange(temp1,temp2,step), color='blue', density=True, alpha = 0.5,label='q_simulated_pow', log=logg)
plt.hist(q, bins=np.arange(temp1,temp2,step), density=True, alpha = 0.5,label='q_observed', histtype='step', color='black', log=logg)
plt.xlabel('q (mm/day)')
plt.ylabel('PDF (-)')
plt.legend(loc='upper right')
plt.show()



# =============================================================================
# 
# #####################
# ## PLOT recessions #2
# #####################
# 
# #compares single recession with the fit 
# 
# t_master           = np.arange(0.1,100,1)
# q_master_recession       = (a*(b-1) * t_master)**(1/(1-b))
# q_master_recession_cloud = (a_cloud*(b_cloud-1) * t_master)**(1/(1-b_cloud))
# 
# # Plot the 2D array against the 1D array
# for i in range(len(q_recessions)):
#     t_0 = q_recessions[i,0]**(1-b) /( a*(b-1))
#     plt.scatter(np.array(range(t_max-1)) +t_0 , q_recessions[i], label='Line {}'.format(i+1),marker='.')
# 
# 
# 
#     plt.plot(t_master,q_master_recession, color='red',label='single recession')
#     plt.plot(t_master,q_master_recession_cloud, color='blue',label='all recessions')
#     # Add labels and legend
#     plt.title('all recessions')
#     plt.xlabel('time(day)')
#     plt.ylabel('q (mm/day)')
#     plt.ylim(np.nanmin(q_recessions), np.nanmax(q_recessions))
#     plt.xscale('log')  # Set the y-axis to use a logarithmic scale
#     plt.yscale('log')  # Set the y-axis to use a logarithmic scale
#     plt.legend(loc='lower left')
#     plt.show()
# 
# =============================================================================



'''
this part extract the hydrograph corresponding to the largest recorded flood and estimates 
the peak discharge and the time between the rising and falling limb have a discharge q=q_max/2
'''


# Boolean array where True indicates a local minimum
local_minima = (q[1:-1] < q[:-2]) & (q[1:-1] < q[2:])
# Indices of local minima
minima_indices = np.where(local_minima)[0] + 1  # +1 to correct the index offset
# Values of local minima
minima_values = q[minima_indices]

#maximum of the timeserie
i_max =  np.argmax(q)
q_max = q[i_max]

i_start  = i_max  -  np.min(i_max-minima_indices[minima_indices<i_max])
i_end    = i_max  +  np.min(minima_indices[minima_indices>i_max] - i_max)
duration = i_end  - i_start

DQ = q[i_max] - q[i_start]
q_extracted = q[i_start:i_end]

#finds the time interval DT between the moment both the increasing and the decreasing limb of the hydrograph are q_max/2
frq = 0.01
t_original       = np.arange(0,duration,1)
t_interpolation  = np.arange(0,duration,frq)

q_extracted_interpolated  = np.interp(t_interpolation, np.arange(0,duration,1), q_extracted)

id_max= np.argmax(q_extracted_interpolated)

difference = np.abs( q_extracted_interpolated - q_max/2)
t_1= np.argmin( difference[0:id_max])
t_2= id_max + np.argmin( difference[id_max:])


DT = ( t_2 - t_1) / frq # goes back to days


plt.plot(t_original,q_extracted)
plt.plot([t_1*frq,t_2*frq], [q_max/2,q_max/2])
plt.show()

plt.plot(q)
plt.scatter(minima_indices, minima_values, marker='.', c='red')
plt.scatter(i_start, q[i_start], marker='.', c='black')
plt.xlim(750, 850)


'''
this part extract ALL hydrographs matching a certain criteria
'''

#threshold q_start (initial discharge)
q_threshold_start  = np.percentile(q,100)

# Boolean array where True indicates a local minimum
local_minima = (q[1:-1] <= q[:-2]) & (q[1:-1] <= q[2:])
local_minima = np.pad(local_minima, pad_width=(1, 1), mode='constant', constant_values=0)

#discard those higher than q_threshold
local_minima[np.where(q[local_minima]>q_threshold_start)] = False

# Values of local minima
q_minima = q[local_minima]

minima_indices=np.where(local_minima)[0]

q_threshold_peak = np.percentile(q, 10)
q_threshold_end  = np.percentile(q, 60)
len_treshold     = 10
H = []
max_len=0
for i in range(len(minima_indices)-1):
    q_hydrograph = q[ minima_indices[i] : minima_indices[i+1]]
    if (np.max(q_hydrograph) > q_threshold_peak): # and
        #q_hydrograph[0] < q_threshold_end and
        #q_hydrograph[-1] < q_threshold_end ):
        
        len_q = len(q_hydrograph) 
        
        if len_q > len_treshold:
            H.append(q_hydrograph)
    
            plt.plot(q_hydrograph)
            #plt.show()

            if len_q > max_len:
                max_len=len(q_hydrograph)    
        

frame= max_len*2
HH = np.full((len(H),frame), np.nan)
position_peak  = int(frame/2)

for i, h in enumerate(H):
    shift = position_peak -  np.argmax(h) 
    HH[i,shift:shift+len(h)] = h[np.newaxis, :]


plt.plot(HH.T)
plt.show()



# Define the customized function to fit
def custom_function(x, a, b):
    return a + b*x

flood_volume = np.nansum(HH,axis=1)
flood_peak   = np.nanmax(HH,axis=1)

# Fit the function to the data
initial_guess = [1, 1.0]  
params, covariance = curve_fit(custom_function, np.log(flood_volume), np.log(flood_peak), p0=initial_guess)
a, b = params

vals = np.linspace(np.min(flood_volume),np.max(flood_volume) )
plt.scatter(flood_volume ,flood_peak, marker = '.')
plt.plot( vals, np.exp(a)*vals**b , color = 'r'  )
plt.xscale('log'),plt.yscale('log')
plt.xlabel('Hydrograph volume (mm)'), plt.ylabel('Hydrograph peak (mm/day)')
textstr = f'y= {np.exp(a):.2f} x ^ {b:.2f}'
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.show()




# =============================================================================
# ## find the width at Q_max/2
# Q_max=[]   
# DT=[]  
# for i, h in enumerate(H):
# 
#     q_max = np.nanmax(h)
#     frq = 0.01
#     duration= len(h)
#     t_original       = np.arange(0,duration,1)
#     t_interpolation  = np.arange(0,duration,frq)
#     h_interpolated  = np.interp(t_interpolation, t_original, h)
#     id_max= np.argmax(h_interpolated)
#     if id_max > 0 :
#         difference = np.abs( h_interpolated - q_max/2)
#         t_1= np.argmin( difference[0:id_max])
#         t_2= id_max + np.argmin( difference[id_max:])
#         dt = ( t_2 - t_1) * frq # goes back to days
#         
#         Q_max.append(q_max)
#         DT.append(dt)
# plt.scatter(DT,Q_max)
# =============================================================================





        
##########################   continua qua....




# =============================================================================
# #maximum of the timeserie
# i_max =  np.argmax(q)
# q_max = q[i_max]
# 
# i_start  = i_max  -  np.min(i_max-minima_indices[minima_indices<i_max])
# i_end    = i_max  +  np.min(minima_indices[minima_indices>i_max] - i_max)
# duration = i_end  - i_start
# 
# DQ = q[i_max] - q[i_start]
# q_extracted = q[i_start:i_end]
# 
# #finds the time interval DT between the moment both the increasing and the decreasing limb of the hydrograph are q_max/2
# frq = 0.01
# t_original       = np.arange(0,duration,1)
# t_interpolation  = np.arange(0,duration,frq)
# 
# q_extracted_interpolated  = np.interp(t_interpolation, np.arange(0,duration,1), q_extracted)
# 
# id_max= np.argmax(q_extracted_interpolated)
# 
# difference = np.abs( q_extracted_interpolated - q_max/2)
# t_1= np.argmin( difference[0:id_max])
# t_2= id_max + np.argmin( difference[id_max:])
# 
# 
# DT = ( t_2 - t_1) / frq # goes back to days
# 
# 
# plt.plot(t_original,q_extracted)
# plt.plot([t_1*frq,t_2*frq], [q_max/2,q_max/2])
# plt.show()
# 
# plt.plot(q)
# plt.scatter(minima_indices, minima_values, marker='.', c='red')
# plt.scatter(i_start, q[i_start], marker='.', c='black')
# plt.xlim(750, 850)
# 
# =============================================================================




## power spectrum

#in frequency domain
# Compute the FFT
fft_values = np.fft.fft(q)
# Compute the power spectrum
power_spectrum = np.abs(fft_values)**2 / len(q)
# Get the frequency bins
freq_bins = np.fft.fftfreq(len(q), d=1)  # d is the sampling interval, typically 1 day for daily data
# Filter positive frequencies for plotting
positive_freq_indices = freq_bins > 0
positive_freq_bins = freq_bins[positive_freq_indices]
positive_power_spectrum = power_spectrum[positive_freq_indices]
# Plot the power spectrum on a log-log scale
plt.figure(figsize=(10, 4))
plt.loglog(positive_freq_bins, positive_power_spectrum)
plt.title('Log-Log Plot of Power Spectrum of Discharge Time Series')
plt.xlabel('Frequency (1/day)')
plt.ylabel('Power')
plt.grid(True, which='both', ls='--')  # Add grid lines for better readability
plt.show()



#in period domain 
# Compute the FFT
fft_values = np.fft.fft(q)
# Compute the power spectrum with normalization
power_spectrum = np.abs(fft_values)**2 / len(q)
# Get the frequency bins
freq_bins = np.fft.fftfreq(len(q), d=1)  # d is the sampling interval, typically 1 day for daily data
# Filter positive frequencies for plotting
positive_freq_indices = freq_bins > 0
positive_freq_bins = freq_bins[positive_freq_indices]
positive_power_spectrum = power_spectrum[positive_freq_indices]
# Convert frequency to period
positive_period_bins = 1 / positive_freq_bins
# Plot the power spectrum on a log-log scale with period on the x-axis
plt.figure(figsize=(10, 4))
plt.loglog(positive_period_bins, positive_power_spectrum, 'o', markersize=1, label='Data Points')
plt.title('Log-Log Plot of Power Spectrum of Discharge Time Series')
plt.xlabel('Period (days)')
plt.ylabel('Power')
plt.grid(True, which='both', ls='--')  # Add grid lines for better readability
plt.show()
