##
import warnings
warnings.filterwarnings("ignore")

RCS0X = "RCS07"

#min_thresh = 1
second_threshold = 30
post_stim_buffer = 10

#! imports
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import datetime
from matplotlib.dates import DateFormatter, AutoDateLocator
import os

#out_dir = "/home/jsaal/s0_ps_an/data/{}/".format(RCS0X)
out_dir = f'/home/jsaal/analysis/network/data/{RCS0X}/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
#@

#!def get_arm1_df(RCS0X):
def get_arm1_df(RCS0X):
    if RCS0X == 'RCS02':
        #redcap_df = pd.read_csv('/home/jsaal/biomarker_pipeline/RCS02/redcap_records_RCS02_and_RCS03_JS.csv')
        redcap_df = pd.read_csv('/home/jsaal/ppt_files/RCS02/redcap_records_RCS02_and_RCS03_JS.csv')
        pt = 2
        start_date = '2020-7-28'
        #remove the last row
        #redcap_df = redcap_df[:-1]
        #breakpoint()
    elif RCS0X == 'RCS04':
        redcap_df = pd.read_csv('/home/jsaal/ppt_files/RCS04/redcap_records_RCS04.csv')
        pt = 4
        start_date = '2021-03-11'
    elif RCS0X == 'RCS05':
        redcap_df = pd.read_csv('/datastore_spirit/human/ChronicPain_NK/redcap_records/redcap_records_RCS05.csv')
        pt = 5
        start_date = '2021-06-09'
    elif RCS0X == 'RCS06':
        redcap_df = pd.read_csv("/datastore_spirit/human/ChronicPain_NK/redcap_records/redcap_records_RCS06.csv")
        start_date = '2022-03-02'
        pt = 6
    elif RCS0X == 'RCS07':
        redcap_df = pd.read_csv('/home/jsaal/ppt_files/RCS07/redcap_records_RCS07.csv')
        start_date = '2022-09-21'
        pt = 7

    #get datetime from 7am onwards
    start_date = pd.to_datetime(start_date) + pd.Timedelta('7 hours')

    redcap_df['pt'] = redcap_df['pt'].astype('float32')
    #select the patients redcap scores
    patient_df = redcap_df[redcap_df['pt'] == pt]
    #select vasmpq rows
    #arm1_df = patient_df[~patient_df['scales_vasmpq_timestamp'].isna()]
    arm1_df = patient_df[~patient_df['stim_onoff_timestamp'].isna()]

    columns_of_interest = [
        "record_id",
        "stim_onoff_timestamp",
        "scales_vasmpq_timestamp",
        "short_vs_long",
        "nrs_s0",
        "intensity_vas_s0",
        "mood_vas_s0",
        "unpleasantness_vas_s0"]

    arm1_df = arm1_df[columns_of_interest]

    #convert the timestamp column to datetime
    arm1_df['stim_onoff_timestamp'] = pd.to_datetime(arm1_df['stim_onoff_timestamp'])
    arm1_df['scales_vasmpq_timestamp'] = pd.to_datetime(arm1_df['scales_vasmpq_timestamp'])


    #select the rows that are after the start date
    arm1_df = arm1_df[arm1_df['stim_onoff_timestamp'] > start_date]

    #if it's RCS02, remove the last row
    if RCS0X == 'RCS02':
        arm1_df = arm1_df[:-1]

    return arm1_df
#@

#!def get_ppt_info(RCS0X):
def get_ppt_info(RCS0X):
    ppt_info_fold = "/home/jsaal/ppt_files/{}/".format(RCS0X)
    stim_info_fname = ppt_info_fold + "{}_stim_info.csv".format(RCS0X)

    #load dataframes for survey and stim info
    stim_info_df = pd.read_csv(stim_info_fname)
    arm1_df = get_arm1_df(RCS0X)

    #add a column to the stim_info_df that is the 'stim_start' converted into datetime
    stim_info_df['stim_start_dt'] = pd.to_datetime(stim_info_df['stim_start'])
    #do the same for 'stim_stop'
    stim_info_df['stim_stop_dt'] = pd.to_datetime(stim_info_df['stim_stop'])

    return stim_info_df, arm1_df
#@

#!def plot_stim_events(stim_info_df, arm1_df):
def plot_stim_events(stim_info_df, arm1_df):
    # Combine the two datetime columns into a single column
    #duplicate_rows = stim_info_df[stim_info_df['stim_start_dt'] == stim_info_df['stim_stop_dt']]
    #stim_info_df = stim_info_df.drop(duplicate_rows.index)

    stim_start_df = stim_info_df[['stim_start_dt']].copy()

    stim_start_df['stim_status'] = 1

    stim_stop_df = stim_info_df[['stim_stop_dt']].copy()
    stim_stop_df['stim_status'] = 0

    stim_df = pd.concat([stim_start_df.rename(columns={'stim_start_dt': 'datetime'}),
                         stim_stop_df.rename(columns={'stim_stop_dt': 'datetime'})])

    # Sort the dataframe by datetime
    stim_df.sort_values('datetime', inplace=True)

    # Create a time range using timedelta
    time_range = pd.date_range(stim_df['datetime'].min(), stim_df['datetime'].max(), freq='S')
    stim_resampled = stim_df.set_index('datetime').reindex(time_range, fill_value=np.nan).fillna(method='ffill')

    # Enable interactive mode
    plt.ion()

    # Create the figure and axis
    fig, ax = plt.subplots()

    # Plot the data
    ax.plot(stim_resampled.index, stim_resampled['stim_status'])

    # Add red vertical lines for each datetime in 'scales_vasmpq_timestamp'

    for timestamp in arm1_df['stim_onoff_timestamp']:
        ax.axvline(timestamp, color='red', linestyle='--', alpha=0.7, linewidth=0.1)

    # Set the x-axis major formatter and locator
    ax.xaxis.set_major_formatter(DateFormatter('%m/%d %H:%M:%S'))
    ax.xaxis.set_major_locator(AutoDateLocator())

    # Set labels and title
    ax.set_xlabel('Datetime')
    ax.set_ylabel('Stimulation Status')
    ax.set_title('Brain Stimulation Events Over Time')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, fontsize=5)

    #remove y ticks
    plt.yticks([])

    # Display the plot
    plt.show()
#@

#!def add_closest_stim(stim_info_df, arm1_df):
def add_closest_stim(stim_info_df, arm1_df):
    # Calculate the closest stim_stop time before each scales_vasmpq_timestamp
    arm1_df['closest_stim_stop'] = arm1_df['stim_onoff_timestamp'].apply(
        lambda x: stim_info_df['stop_adj'][stim_info_df['stop_adj'] <= x].max())

    #get the post contact

    # Calculate the next closest stim_stop time after each scales_vasmpq_timestamp
    arm1_df['next_closest_stim_stop'] = arm1_df['stim_onoff_timestamp'].apply(
        lambda x: stim_info_df['stop_adj'][stim_info_df['stop_adj'] >= x].min())

    # Calculate the time difference between scales_vasmpq_timestamp and closest stim_stop time
    arm1_df['time_diff_to_stim_stop'] = arm1_df['stim_onoff_timestamp'] - arm1_df['closest_stim_stop']

    # Calculate the closest stim_start time after each scales_vasmpq_timestamp
    arm1_df['closest_stim_start'] = arm1_df['stim_onoff_timestamp'].apply(
        lambda x: stim_info_df['start_adj'][stim_info_df['stim_start_dt'] >= x].min())

    # Calculate the time difference between scales_vasmpq_timestamp and closest stim_start time
    arm1_df['time_diff_to_stim_start'] = arm1_df['closest_stim_start'] - arm1_df['stim_onoff_timestamp']

    return arm1_df
#@

#!def plot_histograms(arm1_df):
def plot_histograms(arm1_df):

    total_surveys = len(arm1_df)
    #select only the rows where the time difference to the closest stim_stop is less than 5 minutes
    arm1_df = arm1_df[arm1_df['time_diff_to_stim_stop'] < pd.Timedelta('5 minutes')]
    #select only the rows where the time difference to the closest stim_start is less than 5 minutes
    arm1_df = arm1_df[arm1_df['time_diff_to_stim_start'] < pd.Timedelta('5 minutes')]


    # Plot the histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot the first histogram (time_diff_to_stim_stop)
    ax1.hist(arm1_df['time_diff_to_stim_stop'].astype('timedelta64[s]'), bins=50, color='blue', alpha=0.7)
    ax1.set_xlabel('Time Difference to Closest Stim Stop (seconds)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Time Differences to \nClosest Stim Stop\nTotal Surveys: {}'.format(total_surveys))

    # Plot the second histogram (time_diff_to_stim_start)
    ax2.hist(arm1_df['time_diff_to_stim_start'].astype('timedelta64[s]'), bins=50, color='green', alpha=0.7)
    ax2.set_xlabel('Time Difference to Closest Stim Start (seconds)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Time Differences to \nClosest Stim Start\nTotal Surveys: {}'.format(total_surveys))

    # Display the plots
    plt.show()
    #@

#!def datestr_to_datetime(date_str):
def datestr_to_datetime(date_str):
    return datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S.%f')
#@

#!def get_delay_info():
def get_delay_info(RCS0X):
    df = pd.read_csv('/datastore_spirit/human/ChronicPain_NK/biopac/{}/biopac_lags.csv'.format(RCS0X))
    all_bp_lags = df['lag, (seconds, pos = move edf forward, neg = backward)'].values
    all_bp_starts = df['start'].values
    all_bp_ends = df['end'].values
    all_bp_filenames = df['filename'].values


    bp_dt_starts = []
    bp_dt_ends = []
    for i in range(len(all_bp_starts)):
        bp_dt_starts.append(datestr_to_datetime(all_bp_starts[i]))
        bp_dt_ends.append(datestr_to_datetime(all_bp_ends[i]))


    return all_bp_lags, bp_dt_starts, bp_dt_ends, all_bp_filenames, df
#@

#!def identify_outliers(data, factor=1.5):
def identify_outliers(data, factor=1.5):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - (factor * iqr)
    upper_bound = q3 + (factor * iqr)
    outlier_indices = np.where((data < lower_bound) | (data > upper_bound))
    outlier_values = data[outlier_indices]
    return outlier_indices, outlier_values
#@

#!def get_regression_results(RCS0X):
def get_regression_results(RCS0X):
    all_bp_lags, bp_dt_starts, bp_dt_ends, bp_filenames, delay_df = get_delay_info(RCS0X)

    all_bp_lags = np.array(all_bp_lags)  # Ensure that all_bp_lags is a NumPy array
    outlier_indices, outlier_values = identify_outliers(all_bp_lags)

    #remove outliers from delay_df
    delay_df = delay_df.drop(outlier_indices[0])
    delay_df.columns

    #remove the first column
    df = delay_df.drop(delay_df.columns[0], axis=1)

    #df = delay_df.drop(df.columns[0], axis=1)

    df.columns = ['lag', 'start', 'end', 'filename', 'correlation']
    start_datetime_objects = [datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S.%f') for dt in df['start'].values]
    stop_datetime_objects = [datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S.%f') for dt in df['end'].values]
    #add the start and stop datetime objects to the dataframe
    df['start_datetime'] = start_datetime_objects
    df['stop_datetime'] = stop_datetime_objects
    # convert the start_datetime to a timestamp
    df['timestamp'] = pd.to_datetime(df['start_datetime']).astype(int) // 10**9 #convert to seconds, fixes weirdness
    #including the datetime accounts for the unevenly spaced data
    huber_t = sm.RLM(df['lag'], sm.add_constant(df[['timestamp']]), M=sm.robust.norms.HuberT())
    # fit the model
    regression_results = huber_t.fit()

    #plot the regression results against the lags
    #fig, ax = plt.subplots(figsize=(15, 6))
    #ax.scatter(df['timestamp'], df['lag'], color='blue', alpha=0.7)
    #ax.plot(df['timestamp'], regression_results.fittedvalues, color='red')
    #ax.set_xlabel('Timestamp')
    #ax.set_ylabel('Lag (seconds)')
    #ax.set_title('Lag vs Timestamp')
    #plt.show()



    return regression_results
#@

#!def adjust_stim_times(regression_results, stim_info_df):
def adjust_stim_times(regression_results, stim_info_df):
    #we want to adjust the EDF time backwards, so we subtract the lag from the timestamp
    #for each row in stim info df
    for index, row in stim_info_df.iterrows():
        curr_ts = row['stim_start_dt'].timestamp()
        bp_lag = regression_results.predict([1, curr_ts])[0]
        #add the stim_start_dt pluts the bp_lag to stim_info_df as a new column
        stim_info_df.loc[index, 'bp_lag'] = bp_lag
        stim_info_df.loc[index, 'start_adj'] = row['stim_start_dt'] - datetime.timedelta(seconds=bp_lag)
        stim_info_df.loc[index, 'stop_adj'] = row['stim_stop_dt'] - datetime.timedelta(seconds=bp_lag)
    return stim_info_df
#@

#!def const_adjust_stim_times(RCS0X, stim_info_df):
def const_adjust_stim_times(RCS0X, stim_info_df):
    #we want to adjust the EDF time backwards, so we subtract the lag from the timestamp
    #for each row in stim info df
    for index, row in stim_info_df.iterrows():
        #curr_ts = row['stim_start_dt'].timestamp()
        #bp_lag = regression_results.predict([1, curr_ts])[0]
        #add the stim_start_dt pluts the bp_lag to stim_info_df as a new column
        if RCS0X == 'RCS02':
            bp_lag = 0
        elif RCS0X == 'RCS03':
            bp_lag = 0
        stim_info_df.loc[index, 'bp_lag'] = bp_lag
        stim_info_df.loc[index, 'start_adj'] = row['stim_start_dt'] - datetime.timedelta(seconds=bp_lag)
        stim_info_df.loc[index, 'stop_adj'] = row['stim_stop_dt'] - datetime.timedelta(seconds=bp_lag)
    return stim_info_df
#@

#!def get_spaced_df(arm1_df, stim_info_df, second_threshold, buffer_time):

"""
get a new arm1_df, but with only the rows that are spaced out by at
least X seconds before preceding stim, including a buffer
a buffer essentially changes the time of the survey we are considering
we can bufer it back ten seconds if we don't want to look at the period right at beginning
of the survey
post_stim_buffer_time gives us a buffer before the previos stim ending to allow, for example,
for the data to return to normal after stim, no discharge
"""
def get_spaced_df(arm1_df, stim_info_df, second_threshold, buffer_time,\
        post_stim_buffer_time = 0):
    # Create the buffered_survey_time column
    arm1_df['buffered_survey_time'] = arm1_df['stim_onoff_timestamp'] - \
            pd.Timedelta(seconds=buffer_time)

    # Find the closest stim_stop_dt before each buffered_survey_time
    arm1_df['buff_closest_stim_stop'] = arm1_df['buffered_survey_time'].apply(
        lambda x: stim_info_df['stop_adj'][stim_info_df['stop_adj'] <= x].max())


    # Calculate the next closest stim_stop time after each scales_vasmpq_timestamp
    arm1_df['buff_next_closest_stim_stop'] = arm1_df['buffered_survey_time'].apply(
        lambda x: stim_info_df['stop_adj'][stim_info_df['stop_adj'] >= x].min())


    # Calculate the time difference between buffered_survey_time and closest_stim_stop
    arm1_df['time_diff_to_closest_stim_stop'] = arm1_df['buffered_survey_time'] - \
           arm1_df['buff_closest_stim_stop']


    # Calculate the closest stim_start time after each scales_vasmpq_timestamp
    arm1_df['buff_closest_stim_start'] = arm1_df['buffered_survey_time'].apply(
        lambda x: stim_info_df['start_adj'][stim_info_df['stim_start_dt'] >= x].min())


    # Select rows where the time difference is greater than or equal to second_threshold
    spaced_arm1_df = arm1_df[arm1_df['time_diff_to_closest_stim_stop'] >= \
            pd.Timedelta(seconds=second_threshold + post_stim_buffer_time)]\
            .reset_index(drop=True)

    #remove the trials that are DURING stim
    #select rows in which the next start is less than the next stop
    spaced_arm1_df = spaced_arm1_df[spaced_arm1_df['buff_closest_stim_start'] < \
            spaced_arm1_df['buff_next_closest_stim_stop']].reset_index(drop=True)

    # Print the new dataframe
    return spaced_arm1_df
#@

stim_info_df, arm1_df = get_ppt_info(RCS0X)
print("starting with {} surveys".format(arm1_df.shape[0]))

#remove rows with duplicate 'stim_start' column value
stim_info_df = stim_info_df.drop_duplicates(subset='stim_start', keep='first').reset_index(drop=True)
print("after removing duplicates, we have {} stim events".format(stim_info_df.shape[0]))

#remove rows with duplicate 'stim_start' or 'stim_stop' row value

#remove rows where 'stim_start' == 'stim_stop'
stim_info_df = stim_info_df[stim_info_df['stim_start'] != stim_info_df['stim_stop']].reset_index(drop=True)
print("after removing stim_start == stim_stop, we have {} stim events".format(stim_info_df.shape[0]))

#remove rows in which the 'stim start' appears as a 'stim stop' anywhere in the dataframe
stim_info_df = stim_info_df[~stim_info_df['stim_start'].isin(stim_info_df['stim_stop'])].reset_index(drop=True)
print("after removing stim_start appearing as stim_stop, we have {} stim events".format(stim_info_df.shape[0]))

# Drop any rows with NaN or NaT values
#stim_info_df = stim_info_df.dropna()

# Reset index after dropping rows
#stim_info_df.reset_index(drop=True, inplace=True)


#if RCS0X isn't RCS02 or RCS03, then we need to adjust the stim times
if RCS0X != 'RCS02' and RCS0X != 'RCS03':
    regression_results = get_regression_results(RCS0X)
    stim_info_df = adjust_stim_times(regression_results, stim_info_df)
else:
    stim_info_df = const_adjust_stim_times(RCS0X, stim_info_df)

#save stim_info_df to csv
stim_info_df.to_csv(out_dir + '{}_stim_info_adjusted.csv'.format(RCS0X), index=False)

arm1_df = add_closest_stim(stim_info_df, arm1_df)

#plot_stim_events(stim_info_df, arm1_df)
#arm1_df = arm1_df.dropna()

#plot_histograms(arm1_df)
#remove any rows with 'NaT' values

#min_thresh = 30
#second_threshold = int(min_thresh * 60)
buffer_time = 20

#incldue a buffer time, so we can look a little bit before the first sruvey
spaced_arm1_df = get_spaced_df(arm1_df, stim_info_df, second_threshold, buffer_time,\
        post_stim_buffer_time = post_stim_buffer)
print("using a {} second threshold and {} second buffer, we have {} surveys"\
        .format(second_threshold, post_stim_buffer, spaced_arm1_df.shape[0]))

#spaced_arm1_df.columns

#spaced_arm1_df

#create a new df, with 3 rows, "label" which has "record id" column, "start", \
# which has the buffered survey time minus the second_threshold, and "stop", which is the buffered survey time
df = pd.DataFrame(columns=['label', 'start', 'stop'])
df['label'] = spaced_arm1_df['record_id']
df['start'] = spaced_arm1_df['buffered_survey_time'] - pd.Timedelta(seconds=second_threshold)
df['stop'] = spaced_arm1_df['buffered_survey_time']

#remove the rows in which the time_diff_to_stim_stop is less then 10 seconds passed the threshold
#df = df[df['start'] < spaced_arm1_df['buff_closest_stim_stop']].reset_index(drop=True)

#print("before removing overlapping")
#for t in ['1 minute', '5 minutes', '10 minutes', '15 minutes', '20 minutes', \
#        '25 minutes', '30 minutes']:
#    print("for {}: {} surveys survive".format(t, \
#            sum(arm1_df['time_diff_to_stim_stop'] > pd.Timedelta(t))))

#add a column to df called "adjust" and set all values to 0, or auto
if RCS0X == 'RCS02' or RCS0X == 'RCS03':
    df['adjust'] = 0
else:
    df['adjust'] = 'auto'

#Remove the trials that are overlapping
indices_to_drop = []
for i in range(df.shape[0] - 1):
    time_difference = df['start'][i + 1] - df['stop'][i]
    if time_difference < pd.Timedelta('0 seconds'):
        indices_to_drop.append(i + 1)
        #print("Negative time difference detected at row", i + 1)
        #print("next start:" + str(df['start'][i + 1]))
        #print("current stop:" + str(df['stop'][i]))
df = df.drop(indices_to_drop).reset_index(drop=True)
print("after removing overlapping, we have {} surveys".format(df.shape[0]))


#export to csv
#df.to_csv(out_dir + 'pre_surv_chunk_times_{}s.csv'.format(str(second_threshold)), index=False)
