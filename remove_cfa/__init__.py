#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Stefan Arnau
December 2021
email: arnau@ifado.de

"""

# Imports
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# The function
def remove_cfa(
    X_data_rlocked,
    y_data_rlocked,
    X_data_stimlocked,
    y_data_stimlocked,
    n_layer=8,
    n_neurons=10,
    batch_size=2048,
    n_obs_train=-1,
):

    # For each feature...
    X_all = []
    X_predict = []
    scaler = StandardScaler()
    for f in range(X_data_rlocked.shape[0]):

        # Mean center each trial
        tmp1 = np.apply_along_axis(lambda x: x - x.mean(), 0, X_data_rlocked[f, :, :])
        tmp2 = np.apply_along_axis(
            lambda x: x - x.mean(), 0, X_data_stimlocked[f, :, :]
        )

        # Reshape to trial after trial
        tmp1 = tmp1.reshape(-1, 1, order="F")
        tmp2 = tmp2.reshape(-1, 1, order="F")

        # Scale
        tmp1 = scaler.fit_transform(tmp1)
        tmp2 = scaler.fit_transform(tmp2)

        # Append
        X_all.append(tmp1)
        X_predict.append(tmp2)

    X_train_all = np.squeeze(np.hstack(X_all))
    X_predict = np.squeeze(np.hstack(X_predict))

    # Reshape y-data
    y_train_all = y_data_rlocked.reshape(-1, order="F")

    # Reduce training set size
    if n_obs_train != -1:
        random_idx = np.random.choice(y_train_all.shape[0], n_obs_train, replace=False,)
        X_train_all = X_train_all[random_idx, :].copy()
        y_train_all = y_train_all[random_idx].copy()

    # Split training data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_all, y_train_all, test_size=0.2
    )

    # Specify neural-network using keras API
    activation_function = "elu"
    initializer = "he_normal"
    input_layer = keras.layers.Input(shape=X_train.shape[1], name="input")
    hidden_layers = []
    for hl_idx in range(n_layer):
        if hl_idx == 0:
            inlay = input_layer
        else:
            inlay = hidden_layers[-1]
        hidden_layers.append(
            keras.layers.Dense(
                n_neurons,
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
        "best_model.h5", save_best_only=True
    )
    patience = 10
    earlystop_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience)

    # Fit the model
    history = model.fit(
        [X_train],
        [y_train],
        epochs=1000,
        validation_data=([X_valid], [y_valid]),
        callbacks=[checkpoint_cb, earlystop_cb],
        batch_size=batch_size,
    )

    # Load model
    model = tf.keras.models.load_model("best_model.h5")

    # Make predictions
    y_stimlocked_predicted_1d = model.predict(X_predict)

    # Save dimensionality (n_times, n_trials) of stimlocked data
    stimlocked_dims = y_data_stimlocked.shape

    # Center stimlocked eeg and concatenate trials
    y_data_stimlocked_1d = np.apply_along_axis(
        lambda x: x - x.mean(), 0, y_data_stimlocked
    ).reshape((-1, 1), order="F")

    # Clean
    y_stimlocked_clean_1d = y_data_stimlocked_1d - y_stimlocked_predicted_1d

    # Reshape
    y_data_stimlocked_2d = y_data_stimlocked_1d.reshape(stimlocked_dims, order="F")
    y_stimlocked_predicted_2d = y_stimlocked_predicted_1d.reshape(
        stimlocked_dims, order="F"
    )
    y_stimlocked_clean_2d = y_stimlocked_clean_1d.reshape(stimlocked_dims, order="F")

    return y_data_stimlocked_2d, y_stimlocked_predicted_2d, y_stimlocked_clean_2d