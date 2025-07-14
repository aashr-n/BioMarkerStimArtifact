import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def select_and_categorize_files():
    """Opens a dialog to select up to 20 CSV files and categorizes them."""
    root = tk.Tk()
    root.withdraw()
    filepaths = filedialog.askopenfilenames(
        title="Select up to 20 'stim_info' and 'arm1' CSV files",
        filetypes=[("CSV files", "*.csv")]
    )
    if not filepaths:
        print("No files were selected.")
        return None, None
    if len(filepaths) > 20:
        print(f"Error: You selected {len(filepaths)} files. Please select no more than 20.")
        return None, None

    stim_info_paths, arm1_paths = [], []
    for path in filepaths:
        filename = os.path.basename(path).lower()
        if "stim_info" in filename:
            stim_info_paths.append(path)
        elif "arm1" in filename: # Explicitly check for 'arm1'
            arm1_paths.append(path)
    
    if not stim_info_paths or not arm1_paths:
        print("Error: Could not find at least one 'stim_info' and one 'arm1' file.")
        return None, None
    return stim_info_paths, arm1_paths

def merge_and_sort_data(stim_info_paths, arm1_paths):
    """Reads, merges, sorts, and tags data with a patient ID."""
    all_dfs = []
    try:
        for path in stim_info_paths:
            patient_id = os.path.basename(path)[:5] # Extract Patient ID
            stim_df = pd.read_csv(path)
            stim_df['sort_timestamp'] = pd.to_datetime(stim_df['stop_adj'])
            stim_df['type'] = 'stim'
            stim_df['patient_id'] = patient_id # Add patient ID column
            all_dfs.append(stim_df)
            print(f"✔️ Loaded {os.path.basename(path)} for patient {patient_id}")

        for path in arm1_paths:
            patient_id = os.path.basename(path)[:5] # Extract Patient ID
            arm1_df = pd.read_csv(path)
            arm1_df['sort_timestamp'] = pd.to_datetime(arm1_df['stim_onoff_timestamp'])
            arm1_df['type'] = 'survey'
            arm1_df['patient_id'] = patient_id # Add patient ID column
            all_dfs.append(arm1_df)
            print(f"✔️ Loaded {os.path.basename(path)} for patient {patient_id}")

        if not all_dfs:
            return None

        combined_df = pd.concat(all_dfs, ignore_index=True, sort=False)
        sorted_df = combined_df.sort_values(by='sort_timestamp', ascending=True).reset_index(drop=True)
        
        # Reorder columns to bring important ones to the front
        cols = ['patient_id', 'sort_timestamp', 'type'] + [c for c in sorted_df.columns if c not in ['patient_id', 'sort_timestamp', 'type']]
        return sorted_df[cols]

    except Exception as e:
        print(f"❌ An error occurred during data processing: {e}")
        return None

def analyze_and_plot_stim_survey_pairs(merged_df):
    """Finds stim-survey pairs and generates multi-patient plots."""
    if merged_df is None or merged_df.empty:
        print("Dataframe is empty, cannot analyze pairs.")
        return

    # Find stim events immediately followed by survey events
    is_pair_start = (merged_df['type'] == 'stim') & (merged_df['type'].shift(-1) == 'survey')
    
    # Ensure the stim and survey in a pair belong to the same patient
    same_patient = merged_df['patient_id'] == merged_df['patient_id'].shift(-1)
    is_valid_pair = is_pair_start & same_patient
    
    indices = merged_df[is_valid_pair].index
    if len(indices) == 0:
        print("\nNo consecutive stim-survey pairs for the same patient were found.")
        return

    # Create a DataFrame of the valid pairs
    pair_data = []
    for i in indices:
        time_delta = merged_df.loc[i+1, 'sort_timestamp'] - merged_df.loc[i, 'sort_timestamp']
        pair_data.append({
            'patient_id': merged_df.loc[i, 'patient_id'],
            'time_delta_seconds': time_delta.total_seconds()
        })
    results_df = pd.DataFrame(pair_data)
    
    print(f"\nFound {len(results_df)} total stim-survey pairs across all patients.")

    patients = results_df['patient_id'].unique()
    colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(patients)))

    # --- Plot 1: Stacked Dot Plot (All Patients) ---
    plt.figure(figsize=(14, 8))
    for i, patient in enumerate(patients):
        patient_data = results_df[results_df['patient_id'] == patient]['time_delta_seconds']
        rounded_times = np.round(patient_data)
        time_counts = pd.Series(rounded_times).value_counts()
        plot_x, plot_y = [], []
        for time_val, count in time_counts.items():
            plot_x.extend([time_val] * count)
            plot_y.extend(range(1, count + 1))
        plt.scatter(plot_x, plot_y, color=colors[i], label=patient, alpha=0.7, s=50)
    plt.title('Instances of Time Delays from Stim to Survey (All Patients)')
    plt.xlabel('Time Difference (seconds)')
    plt.ylabel('Instance Count per Second')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    # --- Plot 2: Box Plot (Comparison) ---
    plt.figure(figsize=(10, 7))
    data_to_plot = [results_df[results_df['patient_id'] == p]['time_delta_seconds'] for p in patients]
    bp = plt.boxplot(data_to_plot, labels=patients, patch_artist=True, vert=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    plt.title('Distribution of Time Differences by Patient')
    plt.ylabel('Time Difference (seconds)')
    plt.xlabel('Patient ID')
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # --- Plot 3: Cumulative Frequency Plot (All Patients) ---
    plt.figure(figsize=(14, 8))
    max_time = results_df['time_delta_seconds'].max()
    thresholds = np.arange(0, max_time + 5, 5)
    for i, patient in enumerate(patients):
        patient_deltas = results_df[results_df['patient_id'] == patient]['time_delta_seconds']
        counts_below_threshold = [np.sum(patient_deltas <= t) for t in thresholds]
        plt.plot(thresholds, counts_below_threshold, marker='o', linestyle='-', color=colors[i], label=f'{patient} (Total: {len(patient_deltas)})')
        # Plot a horizontal line at this patient's max count
        plt.axhline(y=len(patient_deltas), color=colors[i], linestyle='--', alpha=0.7)
    plt.title('Cumulative Count of Surveys Below Time Threshold')
    plt.xlabel('Time Difference Threshold (seconds)')
    plt.ylabel('Number of Instances Below Threshold')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    print("\nDisplaying plots. Close plot windows to save the file.")
    plt.show()

def save_file(dataframe):
    """Opens a save dialog and saves the DataFrame to a CSV file."""
    if dataframe is None:
        print("No data to save due to an earlier error.")
        return
    save_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        title="Save Merged CSV File",
        initialfile="MERGED_all_patients_timeline.csv"
    )
    if save_path:
        dataframe.to_csv(save_path, index=False)
        print(f"\n✅ Successfully saved merged data to: {save_path}")
    else:
        print("\nSave operation cancelled.")

def main():
    """Main function to run the complete workflow."""
    stim_files, arm1_files = select_and_categorize_files()
    if stim_files and arm1_files:
        merged_data = merge_and_sort_data(stim_files, arm1_files)
        if merged_data is not None:
            analyze_and_plot_stim_survey_pairs(merged_data)
            save_file(merged_data)

if __name__ == "__main__":
    main()