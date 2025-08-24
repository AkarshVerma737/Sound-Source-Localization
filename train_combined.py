import os
import argparse
import csv
import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import layers, models
import tensorflow as tf
import scipy.io

# ------------------------
# Custom Keras Models
# ------------------------
class NSASMetric(tf.keras.metrics.Metric):
    def __init__(self, d_max, name='nsas', **kwargs):
        super().__init__(name=name, **kwargs)
        self.d_max = tf.constant(d_max, dtype=tf.float32)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
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

class AngularErrorMetric(tf.keras.metrics.Metric):
    def __init__(self, name='angular_error', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
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


class CustomAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='custom_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_class = tf.argmax(y_true, axis=1)
        pred_class = tf.argmax(y_pred, axis=1)
        matches = tf.cast(tf.equal(true_class, pred_class), tf.float32)
        self.correct.assign_add(tf.reduce_sum(matches))
        self.total.assign_add(tf.cast(tf.size(true_class), tf.float32))

    def result(self):
        return self.correct / self.total

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)

def build_classification_model(input_shape):
    input_layer = layers.Input(shape=input_shape) 

    x = input_layer
    for _ in range(3):
        x = layers.Conv2D(filters=8, kernel_size=(2, 2), activation='relu', padding='valid')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(1, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=x)

    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=[CustomAccuracy()])

    return model



# def build_regression_model(input_shape):
#     inputs = keras.Input(shape=input_shape)

#     # 3 Conv2D blocks
#     x = inputs
#     for _ in range(3):
#         x = layers.Conv2D(8, (2, 2), activation='relu', padding='same')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.MaxPooling2D((1, 2))(x)

#     # Dense regression head
#     x = layers.Flatten()(x)
#     x = layers.Dense(128, activation='relu')(x)
#     output = layers.Dense(3, activation='linear')(x)

#     model = keras.Model(inputs=inputs, outputs=output)
#     model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-3), loss= 'mean_squared_error', metrics=['mae', AngularErrorMetric(), NSASMetric(d_max=1.0)])
#     return model

def build_regression_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Spherical Feature Extraction
    x = inputs
    
    # Multi-scale spatial processing
    branch1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    branch2 = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = layers.Concatenate()([branch1, branch2])
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((1, 2))(x)  # Reduce frequency dimension
    
    # Cross-channel attention
    att = layers.GlobalAveragePooling2D()(x)
    att = layers.Dense(32, activation='relu')(att)
    att = layers.Dense(x.shape[-1], activation='sigmoid')(att)
    x = layers.Multiply()([x, att])
    
    # Depthwise separable convolution
    x = layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # ---- Implicit Temporal Aggregation ----
    # Reshape to [batch*38, features] -> will be treated as temporal sequence
    x = layers.Flatten()(x)
    x = layers.Reshape((-1, 128))(x)  # New shape: [batch*38, timesteps, features]
    
    # Bidirectional temporal processing
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.GRU(64))(x)
    
    # ---- Regression Head ----
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(3, activation='tanh')(x)  # Normalized coordinates
    
    model = keras.Model(inputs=inputs, outputs=output)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='mean_squared_error',
        metrics=['mae', AngularErrorMetric(), NSASMetric(d_max=1.0)]
    )
    return model

# ------------------------
# Data Generator
# ------------------------

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_entries, loss, batch_size=64, dim=(16, 109, 2, 38), shuffle=True, data_key='inputfeats_combined'):
        self.orig_dim = dim
        self.dim = (16, 109, 2)
        self.time_steps = 38
        self.batch_size = batch_size
        self.loss = loss
        self.data_entries = data_entries
        self.shuffle = shuffle
        self.data_key = data_key
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data_entries) / self.batch_size))

    def __getitem__(self, index):
        batch_entries = self.data_entries[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(batch_entries)
        #print(f"X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.data_entries)

    def __data_generation(self, batch_entries):
        # Initialize the input tensor
        X = np.empty((self.batch_size * self.time_steps, *self.dim), dtype=np.float32)

        # Initialize the output tensor depending on mode
        if self.loss == 'regression':
            y = np.empty((self.batch_size * self.time_steps, 3), dtype=np.float32)
        elif self.loss == 'classification':
            y = np.empty((self.batch_size * self.time_steps, 10), dtype=np.float32)
        else:
            raise ValueError(f"Unknown loss type: {self.loss}")

        for i, entry in enumerate(batch_entries):
            try:
                mat_data = scipy.io.loadmat(entry[0])
                input_feats = mat_data[self.data_key].astype(np.float32)  # shape: (16, 109, 38)
                #print(f"Loaded shape from .mat: {input_feats.shape}")


                for t in range(self.time_steps):
                    idx = i * self.time_steps + t
                    X[idx, :, :, :] = input_feats[:, :, :, t] 
                    y[idx] = entry[1]  # Same label for all frames (per-file label)

            except Exception as e:
                print(f"Error loading {entry[0]}: {e}")
                for t in range(self.time_steps):
                    idx = i * self.time_steps + t
                    X[idx] = np.zeros((16, 109, 2), dtype=np.float32)
                    y[idx] = entry[1]

        return X, y


# ------------------------
# Read Data Entries
# ------------------------

def read_data_entries(labelpath, data_folder, mode='classification'):
    data_entries = []
    with open(labelpath, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        header = next(csv_reader, None)  # Skip header

        for line in csv_reader:
            # Convert line[0] to string in case it's a number
            matpath = os.path.join(data_folder, str(line[0]))
            if not matpath.endswith('.mat'):
                matpath += '.mat'

            if not os.path.exists(matpath):
                continue  # skip if file doesn't exist

            if mode == 'classification':
                # Convert label vector to list of ints, skipping the first column (filename)
                class_idx = int(line[1])
                label = keras.utils.to_categorical(class_idx, num_classes=10)

            else:  # regression
                # Defensive: check if line has enough elements for regression
                if len(line) < 7:
                    continue
                label = [float(x) for x in line[4:7]]  # x, y, z

            data_entries.append([matpath, label])
    return data_entries

def plot_training_history(history, is_classification=True):
    """
    Plots training history:
    - Always plots loss vs epochs
    - For classification: plots accuracy
    - For regression: plots angular error and NSAS as subplots
    
    Parameters:
        history: Keras History object
        is_classification: True if classification, False for regression
    """
    
    # Plot 1: Loss vs Epochs
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_vs_epochs.png')
    plt.close()

    # Plot 2
    if is_classification:
        # Accuracy Plot
        if 'accuracy' in history.history and 'val_accuracy' in history.history:
            plt.figure(figsize=(8, 5))
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Accuracy vs Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('accuracy_vs_epochs.png')
            plt.close()
    else:
        # Subplot for Angular Error and NSAS
        fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        if 'angular_error' in history.history and 'val_angular_error' in history.history:
            axs[0].plot(history.history['angular_error'], label='Train Angular Error')
            axs[0].plot(history.history['val_angular_error'], label='Validation Angular Error')
            axs[0].set_ylabel('Angular Error (°)')
            axs[0].set_title('Angular Error vs Epochs')
            axs[0].legend()
            axs[0].grid(True)

        if 'nsas' in history.history and 'val_nsas' in history.history:
            # Filter out negative NSAS values (replace with NaN so matplotlib ignores them)
            nsas = np.array(history.history['nsas'])
            val_nsas = np.array(history.history['val_nsas'])

            nsas[nsas < 0] = np.nan
            val_nsas[val_nsas < 0] = np.nan

            axs[1].plot(nsas, label='Train NSAS')
            axs[1].plot(val_nsas, label='Validation NSAS')
            axs[1].set_ylabel('NSAS Score')
            axs[1].set_title('NSAS Score vs Epochs')
            axs[1].legend()
            axs[1].grid(True)

        plt.tight_layout()
        plt.savefig('angular_nsas_vs_epochs.png')
        plt.close()


# ------------------------
# Main Function
# ------------------------

def main():
    parser = argparse.ArgumentParser(prog='train',
                                     description="""Script to train a DOA estimator""")
    parser.add_argument("--input", "-i", required=True, help="Directory where data and labels are", type=str)
    parser.add_argument("--label", "-l", required=True, help="Path to the label csv", type=str)
    parser.add_argument("--output", "-o", default="models", help="Directory to write results", type=str)
    parser.add_argument("--batchsize", "-b", type=int, default=256, help="Choose a batchsize")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs")
    parser.add_argument("--loss", "-lo", type=str, choices=["categorical", "cartesian"],
                        required=True, help="Choose loss representation")

    args = parser.parse_args()
    assert os.path.exists(args.input), "Input folder does not exist!"

    epochs = args.epochs
    batchsize = args.batchsize
    label_path = args.label
    foldername = '{}_batch{}'.format(args.loss, batchsize)

    outpath = os.path.join(args.output, foldername)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    savepath = os.path.join(outpath, 'best_model.{epoch:02d}-{val_loss:.6f}.h5')

    assert os.path.exists(label_path), "Label csv does not exist!"
    mode = 'classification' if args.loss == 'categorical' else 'regression'
    train_data = read_data_entries(label_path, args.input, mode=mode)

    train_data_entries, val_data_entries = train_test_split(train_data, test_size=0.3, random_state=11)

    # Parameters
    params = {'dim': (16, 109, 2, 38),
          'batch_size': batchsize,
          'loss': 'classification' if args.loss == 'categorical' else 'regression',
          'shuffle': True}


    # Generators
    training_generator = DataGenerator(train_data_entries, **params)
    validation_generator = DataGenerator(val_data_entries, **params)

    input_shape = (16, 109, 2)

    if args.loss == 'categorical':
        model = build_classification_model(input_shape)
    else:
        model = build_regression_model(input_shape)

    model.summary()

    # Callbacks
    earlyStopping = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='min')
    mcp_save = ModelCheckpoint(savepath, verbose=1, save_best_only=True, monitor='val_loss', mode='min')

    # Train model
    history = model.fit(training_generator,
                        validation_data=validation_generator,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[earlyStopping, mcp_save])

    xs = history.epoch

    if args.loss == 'categorical':
        plot_training_history(history, is_classification=True)
    else:
        plot_training_history(history, is_classification=False)

if __name__ == "__main__":
    main()
