"""
Minimal helper toolbox for the cleaned decoding pipeline.
Save this file in:
    /home/jsaal/analysis/clean_master_decoding/clean_master_decoding_helper_funcs.py
All other scripts do
    sys.path.append('/home/jsaal/analysis/clean_master_decoding')
    from clean_master_decoding_helper_funcs import *
before use.
"""

import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import nitime.algorithms as tsa                                    # pip install nitime

# --------------------------- configuration --------------------------- #
_cfg_path = None

def set_config_file(path):
    global _cfg_path
    _cfg_path = path

def _cfg():
    if _cfg_path is None:
        raise RuntimeError("Run set_config_file(<yml path>) before calling helpers")
    with open(_cfg_path) as f:
        return yaml.safe_load(f)

# --------------------------- simple getters -------------------------- #
def get_RCS0X():
    return _cfg()['info']['subject']

def get_proj_data_dir():
    return _cfg()['info']['proj_data_dir']

def get_h5_dir():
    return _cfg()['info']['h5_dir']

def get_surv_h5_dir():
    return _cfg()['info']['duringsurv_h5_dir']

def get_ch_len():
    return _cfg()['info']['ch_len']

def get_ppt_dir():
    return _cfg()['info']['ppt_dir'].format(get_RCS0X())

def get_ignore_trials():
    return _cfg()['info'].get('ignore_trials', [])

# ------------------------------ io utils ----------------------------- #
def foldcheck(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path if path.endswith('/') else path + '/'

# --------------------------- signal helpers -------------------------- #
def two_sided_freqs(srate, win_samp):
    return np.linspace(0, int(srate), win_samp, endpoint=False)

def compute_fft(data, fs, TW, L, nfft=None):
    if nfft is None:
        nfft = data.shape[1]
    freqs = np.linspace(0, fs, nfft, endpoint=False)
    data  = data - data.mean(1, keepdims=True)

    tapers, _ = tsa.dpss_windows(data.shape[1], TW, L)
    out = np.zeros((data.shape[0], L, len(freqs)), complex)
    for k in range(L):
        tapered = data * tapers[k][None, :]
        out[:, k, :] = np.fft.fft(tapered, n=nfft) / fs
    return out, freqs, None

def get_power(spec):
    return np.abs(spec) ** 2

def get_mt_info(TW, win_sec, n_tapers='auto'):
    if n_tapers == 'auto':
        L = int(2 * TW - 1)
    else:
        L = int(n_tapers)
    freq_res = 2 * TW / win_sec
    return L, freq_res

# ------------------------ anatomy‑label helpers ---------------------- #
def get_ch_labels(ch_type):
    """Return numpy arrays: short_labels, anat_labels, long_labels"""
    chs_path = os.path.join(get_surv_h5_dir(), 'chs.npy')
    chan_names = np.load(chs_path, allow_pickle=True)

    elec_csv = os.path.join(get_ppt_dir(), f'{get_RCS0X()}_chans.csv')
    elec_df  = pd.read_csv(elec_csv)

    short_l, anat_l, long_l = [], [], []
    for name in chan_names:
        if len(name.split(' '))!=3:
            name = name.split(' ')[0] + ' ' +  name.split(' ')[1][0] + ' ' + name.split(' ')[1][1:]
        long_label = name.split(' ')[0] + ' ' + name.split(' ')[1] + name.split(' ')[2]
        short      = name.split(' ')[1] + name.split(' ')[2].split('-')[0]
        anat_label = elec_df.loc[elec_df['Electrode'] == short, 'Anat_label'].values[0]

        long_l.append(long_label)
        short_l.append(short)
        anat_l.append(anat_label)

    return np.array(short_l), np.array(anat_l), np.array(long_l)

#!def bootstrap_conf_int(Nbs, func, data, percentile):
def bootstrap_conf_int(Nbs, func, data, percentile):
    whole_dat_stat = func(data, np.arange(data.shape[0]))
    boot_dist = np.zeros(np.append(Nbs, whole_dat_stat.shape))
    print('bootstrapping...')
    for i in range(Nbs):
        sample_inds = np.random.randint(0, data.shape[0], data.shape[0])
        boot_dist[i, :, :] = func(data, sample_inds)
    conf_high = np.sort(boot_dist, axis=0)[int(percentile * Nbs), :, :]
    conf_low = np.sort(boot_dist, axis=0)[int((1 - percentile) * Nbs), :, :]
    conf_int = (conf_low, conf_high)
    return whole_dat_stat, boot_dist, conf_int
#@

def get_electrodes(feature, df_chans_filtered):
    parts = feature.split("_")
    hem = parts[0]
    if 'high_gamma' in feature:
        freq_idx = 2
    elif 'low_gamma' in feature:
        freq_idx = 2
    else:
        freq_idx = 1
    freq = parts[-freq_idx:]
    region = parts[1:-freq_idx]
    label = hem
    full_region = region[0] 
    label = label + " " + full_region
    #breakpoint()

    #breakpoint()
    if len(region) > 1:
        for part in region[1:]:
            full_region += f"{part}"
            label += f" {part}"
    freq = "_".join(freq)


    #breakpoint()
    electrodes = df_chans_filtered[df_chans_filtered["Anat_label"] == label]["Electrode"].values

    return electrodes, freq, hem, full_region
