#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import os
import numpy as np
import joblib
from scipy.io import loadmat
import scipy.stats
import mne

# Paths
path_predicted_data = "/mnt/data_heap/ecg_removal/6_predicted_data/"
path_file = "/home/plkn/repos/cfa_removal/"
path_meta = "/mnt/data_heap/ecg_removal/0_meta/"
path_results_specificity_itpc = "/mnt/data_heap/ecg_removal/7_results/itpc/"

# Load channel XY-coordinates
channel_coords = loadmat(os.path.join(path_meta, "channel_coords.mat"))[
    "channel_coords"
][:, 0:2]

# Subject list
subject_list = [
    "VP02",
    "VP03",
    "VP04",
    "VP05",
    "VP06",
    "VP07",
    "VP08",
    "VP09",
    "VP10",
    "VP11",
    "VP12",
    "VP13",
    "VP14",
    "VP15",
    "VP16",
    "VP17",
    "VP18",
    "VP19",
    "VP20",
    "VP21",
    "VP22",
    "VP23",
    "VP24",
    "VP25",
    "VP26",
    "VP27",
    "VP28",
    "VP29",
    "VP30",
    "VP31",
    "VP32",
    "VP33",
    "VP34",
    "VP35",
    "VP36",
    "VP37",
    "VP38",
    "VP39",
    "VP40",
    "VP41",
]

# Channel list
channel_list = range(1, 61)

# Channel names
channel_names = str(
    loadmat(os.path.join(path_meta, "channel_names.mat"))["channel_names"]
)[3:-2].split(" ")

# Load erp times
erp_times_stimlock = (
    loadmat(os.path.join(path_meta, "time_stimlock.mat"))["time_stimlock"]
    .astype("float64")
    .ravel()
)

# Create a basic mne info structure
sfreq = 1000
info = mne.create_info(channel_names, sfreq, ch_types="eeg", verbose=None)

# Create a montage
standard_1020_montage = mne.channels.make_standard_montage("standard_1020")

# Iterate subjects nad create a (n_epochs, n_channels, n_times) array
for s, subject in enumerate(subject_list):

    # Specify out file name
    fn_out = f"{subject}_tf_averages.joblib"
    out_file = os.path.join(path_results_specificity_itpc, fn_out)

    print(f"\nProcessing {subject}...\n")

    # Load trialinfo
    trialinfo = loadmat(os.path.join(path_meta, f"{subject}_tinf.mat"))["tinf"].astype(
        "float64"
    )

    # Get number of trials
    n_epochs_sys = sum(trialinfo[:, 0] == 1)
    n_epochs_dia = sum(trialinfo[:, 0] == 2)

    # Init subject epoch matrices
    epoch_matrix_sys_before = np.zeros(
        (n_epochs_sys, len(channel_list), len(erp_times_stimlock))
    )
    epoch_matrix_sys_predicted = np.zeros(
        (n_epochs_sys, len(channel_list), len(erp_times_stimlock))
    )
    epoch_matrix_sys_after = np.zeros(
        (n_epochs_sys, len(channel_list), len(erp_times_stimlock))
    )
    epoch_matrix_dia_before = np.zeros(
        (n_epochs_dia, len(channel_list), len(erp_times_stimlock))
    )
    epoch_matrix_dia_predicted = np.zeros(
        (n_epochs_dia, len(channel_list), len(erp_times_stimlock))
    )
    epoch_matrix_dia_after = np.zeros(
        (n_epochs_dia, len(channel_list), len(erp_times_stimlock))
    )

    # Skip if file exists already
    if os.path.isfile(out_file):
        continue

    # iterate channels
    for c, channel in enumerate(channel_list):

        # Load stimlocked before
        dat = joblib.load(
            os.path.join(
                path_predicted_data,
                f"stim_locked_y_before_{subject}_channel_{str(channel)}.joblib",
            ),
        )
        epoch_matrix_sys_before[:, c, :] = dat[trialinfo[:, 0] == 1, :] * 1e-6
        epoch_matrix_dia_before[:, c, :] = dat[trialinfo[:, 0] == 2, :] * 1e-6

        # Load stimlocked predicted
        dat = joblib.load(
            os.path.join(
                path_predicted_data,
                f"stim_locked_y_predicted_{subject}_channel_{str(channel)}.joblib",
            ),
        )
        epoch_matrix_sys_predicted[:, c, :] = dat[trialinfo[:, 0] == 1, :] * 1e-6
        epoch_matrix_dia_predicted[:, c, :] = dat[trialinfo[:, 0] == 2, :] * 1e-6

    # Substract predictions
    epoch_matrix_sys_cleaned = epoch_matrix_sys_before - epoch_matrix_sys_predicted
    epoch_matrix_dia_cleaned = epoch_matrix_dia_before - epoch_matrix_dia_predicted

    # Create mne epochs objects
    baseline = (-0.2, 0)
    tmin = erp_times_stimlock[0] / sfreq

    # Epoch matrices to list
    epoch_matrices = [
        epoch_matrix_sys_before,
        epoch_matrix_dia_before,
        epoch_matrix_sys_predicted,
        epoch_matrix_dia_predicted,
        epoch_matrix_sys_cleaned,
        epoch_matrix_dia_cleaned,
    ]

    # For epoch matrix...
    average_tfr_list = []
    for em_idx, em in enumerate(epoch_matrices):

        # Convert to epoch objects
        tmp = mne.EpochsArray(
            em,
            info,
            tmin=tmin,
            baseline=baseline,
        )
        tmp.set_montage(standard_1020_montage)

        # Calculate ITPCs for epochs
        tf_freqs = np.linspace(2, 20, 20)
        tf_cycles = np.linspace(3, 12, 20)
        (ersp, itpc) = mne.time_frequency.tfr_morlet(
            tmp,
            tf_freqs,
            n_cycles=tf_cycles,
            average=True,
            return_itc=True,
            n_jobs=-2,
            decim=5,
        )

        # Prune in time
        itpc.crop(tmin=-0.8, tmax=1)

        # Collect
        average_tfr_list.append(itpc)

    # Save
    joblib.dump(average_tfr_list, out_file)

# Collector lists
ave_tfr_matrices_sys_before = []
ave_tfr_matrices_dia_before = []
ave_tfr_matrices_sys_predicted = []
ave_tfr_matrices_dia_predicted = []
ave_tfr_matrices_sys_cleaned = []
ave_tfr_matrices_dia_cleaned = []
ave_tfr_sys_before = []
ave_tfr_dia_before = []
ave_tfr_sys_predicted = []
ave_tfr_dia_predicted = []
ave_tfr_sys_cleaned = []
ave_tfr_dia_cleaned = []

# Load itpcs for all subjects and reorganize
for s, subject in enumerate(subject_list):

    # Load data. Dims are space x freq x time
    fn_out = f"{subject}_tf_averages.joblib"
    fn = os.path.join(path_results_specificity_itpc, fn_out)
    tmp = joblib.load(fn)
    
    # For now, as it was messed up in tf section. Will be redundant in future computations..
    for idx, _ in enumerate(tmp):
        tmp[idx] = tmp[idx].crop(tmin=-0.8, tmax=1)

    # Reorganize to time × frequencies × space and collect data as matrices
    ave_tfr_matrices_sys_before.append(np.transpose(tmp[0].data, (2, 1, 0)))
    ave_tfr_matrices_dia_before.append(np.transpose(tmp[1].data, (2, 1, 0)))
    ave_tfr_matrices_sys_predicted.append(np.transpose(tmp[2].data, (2, 1, 0)))
    ave_tfr_matrices_dia_predicted.append(np.transpose(tmp[3].data, (2, 1, 0)))
    ave_tfr_matrices_sys_cleaned.append(np.transpose(tmp[4].data, (2, 1, 0)))
    ave_tfr_matrices_dia_cleaned.append(np.transpose(tmp[5].data, (2, 1, 0)))

    # Also collect data as tfr objects
    ave_tfr_sys_before.append(tmp[0])
    ave_tfr_dia_before.append(tmp[1])
    ave_tfr_sys_predicted.append(tmp[2])
    ave_tfr_dia_predicted.append(tmp[3])
    ave_tfr_sys_cleaned.append(tmp[4])
    ave_tfr_dia_cleaned.append(tmp[5])

# Stack matrices
ave_tfr_matrices_sys_before = np.stack(ave_tfr_matrices_sys_before)
ave_tfr_matrices_dia_before = np.stack(ave_tfr_matrices_dia_before)
ave_tfr_matrices_sys_predicted = np.stack(ave_tfr_matrices_sys_predicted)
ave_tfr_matrices_dia_predicted = np.stack(ave_tfr_matrices_dia_predicted)
ave_tfr_matrices_sys_cleaned = np.stack(ave_tfr_matrices_sys_cleaned)
ave_tfr_matrices_dia_cleaned = np.stack(ave_tfr_matrices_dia_cleaned)

# Define adjacency matrix
adjacency, channel_names = mne.channels.find_ch_adjacency(
    ave_tfr_sys_before[0].info, ch_type="eeg"
)

# Plot adjaceny
mne.viz.plot_ch_adjacency(ave_tfr_sys_before[0].info, adjacency, channel_names)

(times, freqs) = ave_tfr_sys_before[0].times, ave_tfr_sys_before[0].freqs

# Define adjacencies for tf-space, and tf only
tfs_adjacency = mne.stats.combine_adjacency(len(freqs), len(times), adjacency)

# We are running an F test, so we look at the upper tail
# see also: https://stats.stackexchange.com/a/73993
tail = 1

# We want to set a critical test statistic (here: F), to determine when
# clusters are being formed. Using Scipy's percent point function of the F
# distribution, we can conveniently select a threshold that corresponds to
# some alpha level that we arbitrarily pick.
alpha_cluster_forming = 0.001

# For an F test we need the degrees of freedom for the numerator
# (number of conditions - 1) and the denominator (number of observations
# - number of conditions):
n_conditions = 2
n_observations = len(subject_list)
df_effect = n_conditions - 1
df_error = n_observations - n_conditions

# Note: we calculate 1 - alpha_cluster_forming to get the critical value
# on the right tail
f_thresh = scipy.stats.f.ppf(1 - alpha_cluster_forming, dfn=df_effect, dfd=df_error)

# Run the cluster based permutation analysis for sys
data1 = ave_tfr_matrices_sys_before
data2 = ave_tfr_matrices_sys_cleaned
cluster_stats_sys = mne.stats.spatio_temporal_cluster_test(
    [data1, data2],
    n_permutations=1000,
    threshold=f_thresh,
    tail=tail,
    n_jobs=-2,
    buffer_size=None,
    adjacency=tfs_adjacency,
    out_type="mask",
)
F_obs_sys, clusters_sys, p_values_sys, _ = cluster_stats_sys

# Run the cluster based permutation analysis for dia
data1 = ave_tfr_matrices_dia_before
data2 = ave_tfr_matrices_dia_cleaned
cluster_stats_dia = mne.stats.spatio_temporal_cluster_test(
    [data1, data2],
    n_permutations=1000,
    threshold=f_thresh,
    tail=tail,
    n_jobs=-2,
    buffer_size=None,
    adjacency=tfs_adjacency,
    out_type="mask",
)
F_obs_dia, clusters_dia, p_values_dia, _ = cluster_stats_dia

# Calculate effect sizes (adjusted partial eta squared, Mordkoff, 2019)
petasq_sys = np.divide((F_obs_sys * df_effect), (F_obs_sys * df_effect) + df_error)
adjpetasq_sys = petasq_sys - ((1 - petasq_sys) * (df_effect / df_error))
petasq_dia = np.divide((F_obs_dia * df_effect), (F_obs_dia * df_effect) + df_error)
adjpetasq_dia = petasq_dia - ((1 - petasq_dia) * (df_effect / df_error))

# Save effect size matrices averaged across channels
fn = os.path.join(path_results_specificity_itpc, "apes_sys.csv")
np.savetxt(fn, adjpetasq_sys.mean((2)).T, delimiter=",")
fn = os.path.join(path_results_specificity_itpc, "apes_dia.csv")
np.savetxt(fn, adjpetasq_dia.mean((2)).T, delimiter=",")

# Save ITPC matrices averaged across subjects and channels
fn = os.path.join(path_results_specificity_itpc, "itpc_sys_before.csv")
np.savetxt(fn, ave_tfr_matrices_sys_before.mean((0, 3)).T, delimiter=",")
fn = os.path.join(path_results_specificity_itpc, "itpc_sys_predicted.csv")
np.savetxt(fn, ave_tfr_matrices_sys_predicted.mean((0, 3)).T, delimiter=",")
fn = os.path.join(path_results_specificity_itpc, "itpc_sys_cleaned.csv")
np.savetxt(fn, ave_tfr_matrices_sys_cleaned.mean((0, 3)).T, delimiter=",")
fn = os.path.join(path_results_specificity_itpc, "itpc_dia_before.csv")
np.savetxt(fn, ave_tfr_matrices_dia_before.mean((0, 3)).T, delimiter=",")
fn = os.path.join(path_results_specificity_itpc, "itpc_dia_predicted.csv")
np.savetxt(fn, ave_tfr_matrices_dia_predicted.mean((0, 3)).T, delimiter=",")
fn = os.path.join(path_results_specificity_itpc, "itpc_dia_cleaned.csv")
np.savetxt(fn, ave_tfr_matrices_dia_cleaned.mean((0, 3)).T, delimiter=",")

# Identify significant clusters
cluster_p = 0.005

# For sys
clu_sig_sys = []
for cl_idx, cl_p in enumerate(p_values_sys):
    if cl_p < cluster_p:
        clu_sig_sys.append(clusters_sys[cl_idx])
        
# Save cluster outlines
for cl_idx, cl_mask in enumerate(clu_sig_sys):
    fn = os.path.join(
        path_results_specificity_itpc, f"cluster_sys_outline_{cl_idx+1}.csv"
    )
    np.savetxt(fn, np.ceil(cl_mask.mean((2)).T), delimiter=",")

# For dia
clu_sig_dia = []
for cl_idx, cl_p in enumerate(p_values_dia):
    if cl_p < cluster_p:
        clu_sig_dia.append(clusters_dia[cl_idx])

# Save cluster outlines
for cl_idx, cl_mask in enumerate(clu_sig_dia):
    fn = os.path.join(
        path_results_specificity_itpc, f"cluster_dia_outline_{cl_idx+1}.csv"
    )
    np.savetxt(fn, np.ceil(cl_mask.mean((2)).T), delimiter=",")
    
# Get evoked objects from effect size matrices, averaged across frequencies, for topo plotting
apes_evoked_sys = mne.EvokedArray(adjpetasq_sys.mean((1)).T / 1000000, ave_tfr_sys_before[0].info, tmin=-0.8)
apes_evoked_dia = mne.EvokedArray(adjpetasq_dia.mean((1)).T / 1000000, ave_tfr_sys_before[0].info, tmin=-0.8)

# Topo times
times = np.arange(-0.8, 0.3, 0.1)

# Plot and save sys topo series
fig = apes_evoked_sys.plot_topomap(ch_type='eeg', times=times, colorbar=False, cmap="Oranges", vmin=-0, vmax=0.6)
fig.savefig(
    os.path.join(
        path_results_specificity_itpc, "topo_apes_sys.png"
    ),
    dpi=300,
    transparent=True,
)

# Plot and save dia topo series
fig = apes_evoked_dia.plot_topomap(ch_type='eeg', times=times, colorbar=False, cmap="Oranges", vmin=-0, vmax=0.6)
fig.savefig(
    os.path.join(
        path_results_specificity_itpc, "topo_apes_dia.png"
    ),
    dpi=300,
    transparent=True,
)

    

  


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
