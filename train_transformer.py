# ===============================
# SOUND SOURCE LOCALIZATION WITH TRANSFORMER
# ===============================
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
import csv
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau



# ===============================
# CONFIGURATION
# ===============================
DATA_FOLDER = "D:/Projects/Learning based Beamforging/SH/Generated Audio/Combined_Octant"
LABEL_CSV = "D:/Projects/Learning based Beamforging/SH/Generated Audio/reg_oct_labels.csv"
SAVE_MODEL_PATH = "D:/Projects/Learning based Beamforging/SH/Output/WIP/best_model.h5"
SAVE_PLOT_DIR = os.path.dirname(SAVE_MODEL_PATH)
EPOCHS = 30
BATCH_SIZE = 20
D_MODEL = 256
NUM_HEADS = 6
FF_DIM = 384
NUM_LAYERS = 4
D_MAX = 1.0

# ===============================
# CUSTOM METRICS
# ===============================
class AngularErrorMetric(tf.keras.metrics.Metric):
    def __init__(self, name='angular_error', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Ensure consistent dtype
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        true_dir = tf.math.l2_normalize(y_true, axis=-1)
        pred_dir = tf.math.l2_normalize(y_pred, axis=-1)
        dot_product = tf.reduce_sum(true_dir * pred_dir, axis=-1)
        angular_error = tf.acos(tf.clip_by_value(dot_product, -1.0, 1.0)) * (180.0 / np.pi)

        self.total.assign_add(tf.reduce_sum(angular_error))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.total / self.count

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


class NSASMetric(tf.keras.metrics.Metric):
    def __init__(self, d_max, name='nsas', **kwargs):
        super().__init__(name=name, **kwargs)
        self.d_max = tf.constant(d_max, dtype=tf.float32)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Ensure consistent dtype
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        true_dir = tf.math.l2_normalize(y_true, axis=-1)
        pred_dir = tf.math.l2_normalize(y_pred, axis=-1)
        dot_product = tf.reduce_sum(true_dir * pred_dir, axis=-1)
        angular_error = tf.acos(tf.clip_by_value(dot_product, -1.0, 1.0)) * (180.0 / np.pi)
        angular_score = 1 - (angular_error / 180.0)

        distance_error = tf.norm(y_true - y_pred, axis=-1)
        distance_score = 1 - (distance_error / self.d_max)

        nsas_values = angular_score * distance_score
        self.total.assign_add(tf.reduce_sum(nsas_values))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.total / self.count

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

# ===============================
# DATA LOADER (integrated)
# ===============================
class SoundSourceLocalizationGenerator(tf.keras.utils.Sequence):
    def __init__(self, label_path, data_folder, batch_size=16, data_key='inputfeats_combined', shuffle=True):
        self.batch_size = batch_size
        self.data_key = data_key
        self.shuffle = shuffle
        self.data = []

        # FIX: Properly open and read CSV
        with open(label_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for line in reader:
                if len(line) < 7:
                    continue
                file_name = line[0]
                mat_path = os.path.join(data_folder, file_name)
                if not os.path.exists(mat_path):
                    print(f"Missing file: {mat_path}")
                    continue
                try:
                    label = [float(x) for x in line[4:7]]
                except ValueError:
                    continue
                self.data.append((mat_path, label))

        print(f"Total valid samples loaded: {len(self.data)}")
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        batch = self.data[index * self.batch_size:(index + 1) * self.batch_size]
        X, Y = [], []
        for mat_path, label in batch:
            try:
                mat = scipy.io.loadmat(mat_path)
                if self.data_key not in mat:
                    print(f"Missing key in {mat_path}")
                    continue
                feat = mat[self.data_key].transpose(3, 0, 1, 2)  # (38, 16, 109, 2)
                X.append(feat)
                Y.append(label)
            except Exception as e:
                print(f"Error loading {mat_path}: {e}")
        return np.array(X), np.array(Y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.data)


# ===============================
# TRANSFORMER MODEL
# ===============================
# def create_transformer_model():
#     inp = layers.Input(shape=(38, 16, 109, 2), name='input_feats')
#     # reshape to (time, tokens, dim)
#     x = layers.Reshape((38, 16*109*2))(inp)
#     # linear projection to D_MODEL
#     x = layers.Dense(D_MODEL)(x)
#     pos = tf.range(start=0, limit=38, delta=1)
#     pos_emb = layers.Embedding(input_dim=38, output_dim=D_MODEL)(pos)
#     x = x + pos_emb
#     # Transformer encoder stack
#     for _ in range(NUM_LAYERS):
#         attn = layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=D_MODEL)(x,x)
#         x = layers.LayerNormalization()(x + attn)
#         ff = layers.Dense(FF_DIM, activation='relu')(x)
#         ff = layers.Dense(D_MODEL)(ff)
#         x = layers.LayerNormalization()(x + ff)
#     # pooling over time
#     x = layers.GlobalAveragePooling1D()(x)
#     out = layers.Dense(3, activation='linear')(x)
#     model = keras.Model(inputs=inp, outputs=out)
#     return model


def create_transformer_model():
    inp = layers.Input(shape=(38, 16, 109, 2), name='input_feats')
    
    # Reshape to (time, token_dim)
    x = layers.Reshape((38, 16 * 109 * 2))(inp)
    
    # Linear projection to D_MODEL
    x = layers.Dense(D_MODEL)(x)

    # Positional embedding
    pos = tf.range(start=0, limit=38, delta=1)
    pos_emb = layers.Embedding(input_dim=38, output_dim=D_MODEL)(pos)
    x = x + pos_emb

    # Transformer encoder stack
    for _ in range(NUM_LAYERS):
        # Multi-head self-attention
        attn_output = layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=D_MODEL)(x, x)
        attn_output = layers.Dropout(0.1)(attn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

        # Feed-forward network
        ff_output = layers.Dense(FF_DIM, activation='relu')(x)
        ff_output = layers.Dense(D_MODEL)(ff_output)
        ff_output = layers.Dropout(0.1)(ff_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff_output)

    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Optional dropout before final output
    x = layers.Dropout(0.1)(x)

    # Output: 3D coordinates
    out = layers.Dense(3, activation='linear')(x)

    model = keras.Model(inputs=inp, outputs=out)
    return model



# ===============================
# TRAIN & EVALUATE
# ===============================
def plot_training_history(history):
    os.makedirs(SAVE_PLOT_DIR, exist_ok=True)

    def save_metric_plot(metric, ylabel, title, filename):
        if metric in history.history:
            val_metric = 'val_' + metric
            if val_metric in history.history:
                plt.figure()
                plt.plot(history.history[metric], label=f'Train {ylabel}')
                plt.plot(history.history[val_metric], label=f'Val {ylabel}')
                plt.title(title)
                plt.xlabel("Epoch")
                plt.ylabel(ylabel)
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(SAVE_PLOT_DIR, filename))
                plt.close()

    # Plot loss
    save_metric_plot('loss', 'Loss', 'Loss vs Epochs', 'loss_curve.png')

    # Plot angular error
    save_metric_plot('angular_error', 'Angular Error (°)', 'Angular Error vs Epochs', 'angular_error.png')

    # Plot NSAS
    save_metric_plot('nsas', 'NSAS Score', 'NSAS Score vs Epochs', 'nsas_score.png')


# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    # Load once
    full_gen = SoundSourceLocalizationGenerator(LABEL_CSV, DATA_FOLDER, batch_size=BATCH_SIZE)
    np.random.shuffle(full_gen.data)

    # Split
    val_split = int(0.8 * len(full_gen.data))
    train_data = full_gen.data[:val_split]
    val_data = full_gen.data[val_split:]

    # Assign to new generators
    train_gen = SoundSourceLocalizationGenerator(LABEL_CSV, DATA_FOLDER, batch_size=BATCH_SIZE)
    val_gen = SoundSourceLocalizationGenerator(LABEL_CSV, DATA_FOLDER, batch_size=BATCH_SIZE)
    train_gen.data = train_data
    val_gen.data = val_data

    # Build & compile model
    model = create_transformer_model()
    model.compile(optimizer='adam', loss='mse',
                  metrics=[AngularErrorMetric(), NSASMetric(D_MAX)])

    print("Train samples:", len(train_gen.data))
    print("Val samples:", len(val_gen.data))
    print("Train batches (len):", len(train_gen))
    print("Val batches (len):", len(val_gen))

    # Sanity check on batches
    x_train, y_train = train_gen[0]
    x_val, y_val = val_gen[0]
    print("Train batch X shape:", x_train.shape)
    print("Train batch Y shape:", y_train.shape)
    print("Val batch X shape:", x_val.shape)
    print("Val batch Y shape:", y_val.shape)

    
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
    early_stop_cb = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1)
    
    # Save only best model
    checkpoint_by_loss = ModelCheckpoint(
        filepath=os.path.join(SAVE_PLOT_DIR, "best_by_val_loss.keras"),
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )

    checkpoint_by_angle = ModelCheckpoint(
        filepath=os.path.join(SAVE_PLOT_DIR, "best_by_val_angle.keras"),
        monitor='val_angular_error',
        mode='min',
        save_best_only=True,
        verbose=1
    )

    # Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks = [checkpoint_by_loss, checkpoint_by_angle, early_stop_cb, lr_schedule]
    )

    # Plot loss/metrics
    plot_training_history(history)
