#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------------------
# This script performs a neural network regression to remove the cardiac field artefact
# from the EEG.
# -----------------------------------------------------------------------------------------

# Imports
import os
from scipy.io import loadmat
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed, dump

# Datapaths
path_meta = "add_path_here"  # Trialinfo, chanlocs, times
path_in = "add_path_here"  # Prepared data eeg and ecg
path_models = "add_path_here"  # Output path for trained models
model_eval_path = "add_path_here"  # Evaluation results
path_predicted_data = "add_path_here"  # Evaluation results

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

# Set hyperparameters
hyperparameters = {
    "use_phase": True,
    "ecg_elec": "both",
    "train_size": 0.25,
    "n_layer": 8,
    "n_neurons": 10,
    "batch_size": 128,
}

# A list for fails
list_failed = []

# Load erp times
times_rlock = (
    loadmat(os.path.join(path_meta, "time_rlock.mat"))["time_rlock"]
    .astype("float64")
    .ravel()
)
times_stimlock = (
    loadmat(os.path.join(path_meta, "time_stimlock.mat"))["time_stimlock"]
    .astype("float64")
    .ravel()
)

# Build a todo-list
todo_list = []
for subject in subject_list:
    for channel in np.arange(1, 61):
        sc_comb = {}
        sc_comb["subject"] = subject
        sc_comb["channel"] = channel
        todo_list.append(sc_comb)

# Function that trains the models
def train_model(model_todo):

    # Unpack
    subj = model_todo["subject"]
    chan = model_todo["channel"]

    # Load training data (trials as rows)
    X_ecg_dropdown = (
        np.asarray(loadmat(os.path.join(path_in, f"{subj}_train_X"))["train_X"])
        .astype("float64")[0, :, :]
        .T
    )
    X_ecg_proper = (
        np.asarray(loadmat(os.path.join(path_in, f"{subj}_train_X"))["train_X"])
        .astype("float64")[1, :, :]
        .T
    )
    X_cycle = (
        np.asarray(loadmat(os.path.join(path_in, f"{subj}_train_X"))["train_X"])
        .astype("float64")[2, :, :]
        .T
    )
    X_rlat = (
        np.asarray(loadmat(os.path.join(path_in, f"{subj}_train_X"))["train_X"])
        .astype("float64")[3, :, :]
        .T
    )
    y = (
        np.asarray(
            loadmat(os.path.join(path_in, f"{subj}_chan{str(chan)}_train_y",))[
                "train_y"
            ]
        )
        .astype("float64")
        .T
    )

    # Remove last 5 trials due to missing values in cardiac phase info
    X_ecg_dropdown = X_ecg_dropdown[:-5, :].copy()
    X_ecg_proper = X_ecg_proper[:-5, :].copy()
    X_cycle = X_cycle[:-5, :].copy()
    X_rlat = X_rlat[:-5, :].copy()
    y = y[:-5, :].copy()

    # Preserve 20 percent of trials as test data
    random_idx = np.random.choice(
        y.shape[0], int(np.floor(y.shape[0] * 0.2)), replace=False
    )
    X_ecg_dropdown_test = X_ecg_dropdown.copy()[random_idx, :]
    X_ecg_proper_test = X_ecg_proper.copy()[random_idx, :]
    X_cycle_test = X_cycle.copy()[random_idx, :]
    X_rlat_test = X_rlat.copy()[random_idx, :]
    y_test = y.copy()[random_idx, :]
    X_ecg_dropdown_train_full = np.delete(X_ecg_dropdown, random_idx, axis=0)
    X_ecg_proper_train_full = np.delete(X_ecg_proper, random_idx, axis=0)
    X_cycle_train_full = np.delete(X_cycle, random_idx, axis=0)
    X_rlat_train_full = np.delete(X_rlat, random_idx, axis=0)
    y_train_full = np.delete(y, random_idx, axis=0)

    # Clean up
    del X_ecg_dropdown
    del X_ecg_proper
    del X_cycle
    del X_rlat
    del y

    # Get number of epochs and time points per epoch in test data
    n_epochs_test = X_ecg_dropdown_test.shape[0]
    n_times_test = X_ecg_dropdown_test.shape[1]

    # Center single trials, convert to 1d, scale
    def prepare_feature(feature):
        scaler = StandardScaler()
        return scaler.fit_transform(
            np.apply_along_axis(lambda x: x - x.mean(), 1, feature).reshape(-1, 1)
        )

    # Center single trials, convert to 1d, scale
    X_ecg_dropdown_test = prepare_feature(X_ecg_dropdown_test)
    X_ecg_proper_test = prepare_feature(X_ecg_proper_test)
    X_cycle_test = prepare_feature(X_cycle_test)
    X_rlat_test = prepare_feature(X_rlat_test)
    X_ecg_dropdown_train_full = prepare_feature(X_ecg_dropdown_train_full)
    X_ecg_proper_train_full = prepare_feature(X_ecg_proper_train_full)
    X_cycle_train_full = prepare_feature(X_cycle_train_full)
    X_rlat_train_full = prepare_feature(X_rlat_train_full)

    # Choose ecg data
    if hyperparameters["ecg_elec"] == "dropdown":
        X_ecg_test = X_ecg_dropdown_test
        X_ecg_train_full = X_ecg_dropdown_train_full
    elif hyperparameters["ecg_elec"] == "proper":
        X_ecg_test = X_ecg_proper_test
        X_ecg_train_full = X_ecg_proper_train_full
    elif hyperparameters["ecg_elec"] == "both":
        X_ecg_test = np.concatenate((X_ecg_dropdown_test, X_ecg_proper_test), axis=1)
        X_ecg_train_full = np.concatenate(
            (X_ecg_dropdown_train_full, X_ecg_proper_train_full), axis=1
        )

    # Concatenate X-data
    if hyperparameters["use_phase"] == True:
        X_train_full = np.concatenate(
            (X_ecg_train_full, X_cycle_train_full, X_rlat_train_full), axis=1,
        )
        X_test = np.concatenate((X_ecg_test, X_cycle_test, X_rlat_test), axis=1,)
    elif hyperparameters["use_phase"] == False:
        X_train_full = X_ecg_train_full
        X_test = X_ecg_test

    # Clean up
    del X_ecg_dropdown_test
    del X_ecg_proper_test
    del X_cycle_test
    del X_rlat_test
    del X_ecg_dropdown_train_full
    del X_ecg_proper_train_full
    del X_cycle_train_full
    del X_rlat_train_full

    # Flatten y-data
    y_train_full = y_train_full.ravel()
    y_test = y_test.ravel()

    # Reduce training set size
    random_idx = np.random.choice(
        int(y_train_full.shape[0]),
        int(np.floor(y_train_full.shape[0] * hyperparameters["train_size"])),
        replace=False,
    )
    X_train_full = X_train_full[random_idx, :]
    y_train_full = y_train_full[random_idx]

    # Split training data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, test_size=0.2
    )

    # Clean up
    del X_train_full
    del y_train_full

    # Specify neural-network via keras functional API
    activation_function = "elu"
    initializer = "he_normal"
    input_layer = keras.layers.Input(shape=X_train.shape[1], name="input")
    hidden_layers = []
    for hl_idx in range(hyperparameters["n_layer"]):
        if hl_idx == 0:
            inlay = input_layer
        else:
            inlay = hidden_layers[-1]
        hidden_layers.append(
            keras.layers.Dense(
                hyperparameters["n_neurons"],
                activation=activation_function,
                kernel_initializer=initializer,
                name=f"hidden_{hl_idx}",
            )(inlay)
        )
    output_layer = keras.layers.Dense(1, name="output")(hidden_layers[-1])

    # Compile the model
    model = keras.Model(inputs=[input_layer], outputs=[output_layer])
    model.summary()
    optimizer = tf.keras.optimizers.Nadam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, name="Nadam",
    )
    model.compile(
        loss=["mean_squared_error"], optimizer=optimizer,
    )

    # Define callbacks
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        f"{path_models}ecg_prediction_model_{subj}_chan_{str(chan)}.h5",
        save_best_only=True,
    )
    patience = 10
    earlystop_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience)

    # Fit the model
    t1 = time.time()
    history = model.fit(
        [X_train],
        [y_train],
        epochs=1000,
        validation_data=([X_valid], [y_valid]),
        callbacks=[checkpoint_cb, earlystop_cb],
        batch_size=hyperparameters["batch_size"],
    )

    # Clean up
    del X_train
    del X_valid
    del y_train
    del y_valid

    # Take time
    training_time = round(time.time() - t1, 2)

    # Some metrics
    train_loss = history.history["loss"][-(patience + 1)]
    val_loss = history.history["val_loss"][-(patience + 1)]
    n_training_epochs = len(history.history["val_loss"])
    time_per_epoch = training_time / n_training_epochs

    # Clean up
    del history

    # Load best model
    model = tf.keras.models.load_model(
        f"{path_models}ecg_prediction_model_{subj}_chan_{str(chan)}.h5"
    )

    # Prediction on test data
    y_predicted = model.predict(X_test).ravel()

    # Clean up
    del X_test

    # Calculate and save test loss
    test_loss = mean_squared_error(y_test, y_predicted)

    # Calculate residual
    y_residual = y_test - y_predicted

    # Save r-locked predictions as 2d data (trials x times)
    dump(
        y_predicted.reshape((n_epochs_test, n_times_test)),
        os.path.join(
            path_predicted_data,
            f"r_locked_y_predicted_{subj}_channel_{str(chan)}.joblib",
        ),
    )

    # Save r-locked ydata as 2d data (trials x times)
    dump(
        y_test.reshape((n_epochs_test, n_times_test)),
        os.path.join(
            path_predicted_data, f"r_locked_y_before_{subj}_channel_{str(chan)}.joblib"
        ),
    )

    # Calculate erps on r-locked data
    erp_rlock_before = y_test.reshape((n_epochs_test, n_times_test)).mean(axis=0)
    erp_rlock_after = y_residual.reshape((n_epochs_test, n_times_test)).mean(axis=0)
    erp_rlock_predicted = y_predicted.reshape((n_epochs_test, n_times_test)).mean(
        axis=0
    )

    # Load stimlocked data (trials as rows)
    X_stimlock_ecg_dropdown = (
        np.asarray(loadmat(os.path.join(path_in, f"{subj}_predi_X"))["predi_X"])
        .astype("float64")[0, :, :]
        .T
    )
    X_stimlock_ecg_proper = (
        np.asarray(loadmat(os.path.join(path_in, f"{subj}_predi_X"))["predi_X"])
        .astype("float64")[1, :, :]
        .T
    )
    X_stimlock_cycle = (
        np.asarray(loadmat(os.path.join(path_in, f"{subj}_predi_X"))["predi_X"])
        .astype("float64")[2, :, :]
        .T
    )
    X_stimlock_rlat = (
        np.asarray(loadmat(os.path.join(path_in, f"{subj}_predi_X"))["predi_X"])
        .astype("float64")[3, :, :]
        .T
    )
    y_stimlock = (
        np.asarray(
            loadmat(os.path.join(path_in, f"{subj}_chan{str(chan)}_predi_y",))[
                "predi_y"
            ]
        )
        .astype("float64")
        .T
    )

    # Get number of epochs and time points per epoch in stimlocked data
    n_epochs_stimlock = X_stimlock_ecg_dropdown.shape[0]
    n_times_stimlock = X_stimlock_ecg_dropdown.shape[1]

    # Center single trials, convert to 1d, scale
    X_stimlock_ecg_dropdown = prepare_feature(X_stimlock_ecg_dropdown)
    X_stimlock_ecg_proper = prepare_feature(X_stimlock_ecg_proper)
    X_stimlock_cycle = prepare_feature(X_stimlock_cycle)
    X_stimlock_rlat = prepare_feature(X_stimlock_rlat)

    # Choose ecg data
    if hyperparameters["ecg_elec"] == "dropdown":
        X_stimlock_ecg = X_stimlock_ecg_dropdown
    elif hyperparameters["ecg_elec"] == "proper":
        X_stimlock_ecg = X_stimlock_ecg_proper
    elif hyperparameters["ecg_elec"] == "both":
        X_stimlock_ecg = np.concatenate(
            (X_stimlock_ecg_dropdown, X_stimlock_ecg_proper), axis=1
        )

    # Compile X-dat
    if hyperparameters["use_phase"] == True:
        X_stimlock = np.concatenate(
            (X_stimlock_ecg, X_stimlock_cycle, X_stimlock_rlat), axis=1,
        )
    elif hyperparameters["use_phase"] == False:
        X_stimlock = X_stimlock_ecg

    # Clean up
    del X_stimlock_ecg_dropdown
    del X_stimlock_ecg_proper
    del X_stimlock_cycle
    del X_stimlock_rlat

    # Center y-data
    y_stimlock = np.apply_along_axis(lambda x: x - x.mean(), 1, y_stimlock).ravel()

    # Predict
    y_stimlock_predicted = model.predict(X_stimlock)

    # Save stim-locked predictions as 2d data (trials x times)
    dump(
        y_stimlock_predicted.reshape((n_epochs_stimlock, n_times_stimlock)),
        os.path.join(
            path_predicted_data,
            f"stim_locked_y_predicted_{subj}_channel_{str(chan)}.joblib",
        ),
    )

    # Save stim-locked ydata as 2d data (trials x times)
    dump(
        y_stimlock.reshape((n_epochs_stimlock, n_times_stimlock)),
        os.path.join(
            path_predicted_data,
            f"stim_locked_y_before_{subj}_channel_{str(chan)}.joblib",
        ),
    )

    # Clean up
    del X_stimlock

    # Calculate test loss
    stimlock_loss = mean_squared_error(y_stimlock, y_stimlock_predicted.ravel())

    # Calculate residual
    y_stimlock_residual = y_stimlock - y_stimlock_predicted.ravel()

    # Calculate erps on stimulus-locked data
    erp_stimlock_before = y_stimlock.reshape(
        (n_epochs_stimlock, n_times_stimlock)
    ).mean(axis=0)
    erp_stimlock_after = y_stimlock_residual.reshape(
        (n_epochs_stimlock, n_times_stimlock)
    ).mean(axis=0)
    erp_stimlock_predicted = y_stimlock_predicted.reshape(
        (n_epochs_stimlock, n_times_stimlock)
    ).mean(axis=0)

    # Result dict
    model_eval_scores = {}
    model_eval_scores["dataset_subject"] = subj
    model_eval_scores["dataset_channel"] = chan
    model_eval_scores["parameter_use_phase"] = hyperparameters["use_phase"]
    model_eval_scores["parameter_ecg_elec"] = hyperparameters["ecg_elec"]
    model_eval_scores["parameter_train_size"] = hyperparameters["train_size"]
    model_eval_scores["parameter_n_layer"] = hyperparameters["n_layer"]
    model_eval_scores["parameter_n_neurons"] = hyperparameters["n_neurons"]
    model_eval_scores["parameter_batch_size"] = hyperparameters["batch_size"]
    model_eval_scores["training_training_time"] = training_time
    model_eval_scores["loss_training"] = train_loss
    model_eval_scores["loss_validation"] = val_loss
    model_eval_scores["training_n_training_epochs"] = n_training_epochs
    model_eval_scores["training_time_per_epoch"] = time_per_epoch
    model_eval_scores["loss_test"] = test_loss
    model_eval_scores["erp_rlock_before"] = erp_rlock_before
    model_eval_scores["erp_rlock_after"] = erp_rlock_after
    model_eval_scores["erp_rlock_predicted"] = erp_rlock_predicted
    model_eval_scores["loss_stimlock"] = stimlock_loss
    model_eval_scores["erp_stimlock_before"] = erp_stimlock_before
    model_eval_scores["erp_stimlock_after"] = erp_stimlock_after
    model_eval_scores["erp_stimlock_predicted"] = erp_stimlock_predicted
    model_eval_scores["erp_times_rlock"] = times_rlock
    model_eval_scores["erp_times_stimlock"] = times_stimlock

    # Save result
    dump(
        model_eval_scores,
        os.path.join(
            model_eval_path,
            f"prediction_model_eval_scores_{subj}_channel_{str(chan)}.joblib",
        ),
    )
    return "success"


# Iterate model list and train
parallel_dump = Parallel(n_jobs=-2)(
    delayed(train_model)(model_todo) for model_todo in todo_list
)

