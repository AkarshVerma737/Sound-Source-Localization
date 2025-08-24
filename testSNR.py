import os
import numpy as np
import scipy.io
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIGURATION ===
MODEL_PATH = 'D:/Projects/Learning based Beamforging/SH/Output/Transformer/best_by_val_angle.keras'
BASE_SNR_DIR = 'D:/Projects/Learning based Beamforging/SH/Generated Audio/TESTDATA'
LABEL_CSV = 'D:/Projects/Learning based Beamforging/SH/Generated Audio/TESTDATA/test_labels.csv'
SNR_LEVELS = [5, 8, 11, 14, 17, 20]  # SNRs in dB

# === CUSTOM METRICS ===
class NSASMetric(tf.keras.metrics.Metric):
    def __init__(self, d_max=1.0, name='nsas', **kwargs):
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

# === LOAD & STACK TEST DATA ===
def load_and_stack_test_data(test_dir):
    mat_files = [f for f in os.listdir(test_dir) if f.endswith('.mat')]
    all_data = []
    all_filenames = []

    for f in sorted(mat_files):
        path = os.path.join(test_dir, f)
        mat = scipy.io.loadmat(path)

        if 'inputfeats_combined' not in mat:
            raise KeyError(f"'inputfeats_combined' not found in {f}. Keys: {mat.keys()}")

        features = mat['inputfeats_combined']  # (16, 109, 2, 38)
        features = np.transpose(features, (3, 0, 1, 2))  # → (38, 16, 109, 2)
        all_data.append(features)
        all_filenames.append(f)

    stacked = np.stack(all_data, axis=0)  # (N, 38, 16, 109, 2)
    return stacked.astype(np.float32), all_filenames

# === COMPUTE AZIMUTH RMSE ===
def compute_azimuth_rmse(preds, filelist, label_df, snr_level):
    rows = []
    for i, fname in enumerate(filelist):
        pred_x, pred_y, pred_z = preds[i]
        pred_phi = np.degrees(np.arctan2(pred_y, pred_x)) % 360  # azimuth in degrees [0, 360)

        rows.append({
            'FileName': os.path.basename(fname),
            'Pred_x': pred_x,
            'Pred_y': pred_y,
            'Pred_z': pred_z,
            'Pred_phi': pred_phi
        })

    result_df = pd.DataFrame(rows)
    result_df['FileName'] = result_df['FileName'].apply(os.path.basename)
    label_df['FileName'] = label_df['FileName'].apply(os.path.basename)

    merged_df = pd.merge(result_df, label_df, on='FileName', how='left')

    # Save predictions
    merged_df[['FileName', 'Pred_x', 'Pred_y', 'Pred_z', 'Pred_phi', 'theta', 'phi']].to_csv(
        f"predictions_SNR_{snr_level}.csv", index=False)

    # Compute azimuth RMSE (degrees)
    true_phi = merged_df['phi'].values % 360
    pred_phi = merged_df['Pred_phi'].values % 360
    angular_errors = np.abs(pred_phi - true_phi)
    angular_errors = np.minimum(angular_errors, 360 - angular_errors)
    rmse = np.sqrt(np.mean(angular_errors ** 2))
    return rmse

# === MAIN ===
def main():
    # Load model
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            'NSASMetric': NSASMetric,
            'AngularErrorMetric': AngularErrorMetric
        }
    )
    print("✅ Model loaded.")

    # Load ground truth labels
    label_df = pd.read_csv(LABEL_CSV, usecols=["FileName", "theta", "phi"])
    label_df['FileName'] = label_df['FileName'].apply(os.path.basename)

    snr_errors = []

    for snr in SNR_LEVELS:
        test_dir = os.path.join(BASE_SNR_DIR, f"SNR{snr}")
        print(f"\n📁 Processing SNR {snr} dB...")

        x_test, filelist = load_and_stack_test_data(test_dir)
        print(f"📦 Test data shape: {x_test.shape}")

        preds = model.predict(x_test, batch_size=32, verbose=0)
        print(f"✅ Predictions done for SNR {snr}.")

        rmse = compute_azimuth_rmse(preds, filelist, label_df.copy(), snr)
        snr_errors.append((snr, rmse))
        print(f"📐 Azimuth RMSE @ {snr} dB = {rmse:.2f}°")

    # === PLOT RMSE vs SNR ===
    snrs, errors = zip(*snr_errors)
    plt.figure(figsize=(8, 5))
    plt.ylim(0, 20)
    plt.plot(snrs, errors, marker='o', linewidth=2, color='blue')
    plt.title("Azimuth RMSE vs SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Azimuth RMSE (degrees)")
    plt.grid(True)
    plt.xticks(snrs)
    plt.tight_layout()
    plt.savefig("azimuth_rmse_vs_snr.png")
    plt.show()
    print("\n📊 Plot saved as azimuth_rmse_vs_snr2.png")

if __name__ == "__main__":
    main()
