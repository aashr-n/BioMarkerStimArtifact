#legacy


import tkinter as tk
from tkinter import filedialog

import pandas as pd

import numpy as np
import datetime

#threshold in seconds, anything above will be not considered (equal will be considered)
TimeThreshold = 300

buffer = 10 #seconds
duration = 30 #seconds


root = tk.Tk()
root.withdraw()  # Hide the main window
filepath = filedialog.askopenfilename(
    title="Select one CSV file of the merged ",
    filetypes=[("CSV files", "*.csv")]
    )

merged_df = pd.read_csv(filepath)



#maybe should round here actually
is_stim_followed_by_survey = (merged_df['type'] == 'stim') & (merged_df['type'].shift(-1) == 'survey')
StimEnd_times = merged_df.loc[is_stim_followed_by_survey, 'sort_timestamp']
StimStart_times = merged_df.loc[is_stim_followed_by_survey, 'start_adj'] #getting whole seconds for some reason

survey_times = merged_df.loc[is_stim_followed_by_survey.shift(1, fill_value=False), 'sort_timestamp']


if StimEnd_times.empty:
    print("\nNo consecutive stim-survey pairs were found in the data.")
    raise SystemExit

StimEnd_times.reset_index(drop=True, inplace=True)
survey_times.reset_index(drop=True, inplace=True)
StimStart_times.reset_index(drop=True, inplace=True)


print(f"First back to back:\nStimulation Start time:{StimStart_times[0]}\nEnd time: {StimEnd_times[0]}\nSurvey time: {survey_times[0]}")

# When read from a CSV, timestamp columns are often strings. They need to be converted to datetime objects for calculations.
# The original code used `datetime.datetime.strptime`, which is designed for a single string, not a pandas Series.
# This would cause a TypeError. The correct, vectorized function to use is `pd.to_datetime`.
survey_times = pd.to_datetime(survey_times)
StimEnd_times = pd.to_datetime(StimEnd_times)


time_deltas = survey_times - StimEnd_times
time_deltas_seconds = time_deltas.dt.total_seconds()


# It's more efficient and idiomatic in pandas to perform this comparison directly
# on the Series. This vectorized operation is much faster than a Python loop.
thresholdMatched = time_deltas_seconds <= TimeThreshold


print(f"\nFound {thresholdMatched.sum()} pairs within the {TimeThreshold} second threshold.")




# makes arrays thresholdMatched.num rows long and 2 columns each
beforeStimRanges = np.zeros((thresholdMatched.sum(), 2), dtype=datetime.datetime)
duringStimRanges = np.zeros((thresholdMatched.sum(), 2), dtype=datetime.datetime)
afterStimRanges = np.zeros((thresholdMatched.sum(), 2), dtype=datetime.datetime)


#make three different .csv files "before", "during" , "after" stims
#shape: each row is a stim instance. each instance has a "start time" and "end time" column.
# the times are in datetime.datetime format


# the during stim range is just the middle "duration" seconds between the start and end of stim
#before stim ranges is the "duration" seconds of time before the buffer before the start
#after stim ranges is the "duration" seconds of time after the buffer after the start