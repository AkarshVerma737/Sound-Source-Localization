import os
import numpy as np
import scipy.io
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIGURATION ===
MODEL_PATH = 'D:/Projects/Learning based Beamforging/SH/Output/Regression/cartesian_batch10/best_model.44-0.004050.h5'
BASE_SNR_DIR = 'D:/Projects/Learning based Beamforging/SH/Generated Audio/TESTDATA'
LABEL_CSV = 'D:/Projects/Learning based Beamforging/SH/Generated Audio/TESTDATA/test_labels.csv'
SNR_LEVELS = [5, 8, 11, 14, 17, 20]

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

# === UTILS ===
def load_and_stack_test_data(test_dir):
    mat_files = [f for f in os.listdir(test_dir) if f.endswith('.mat')]
    all_data = []
    all_filenames = []

    for f in mat_files:
        path = os.path.join(test_dir, f)
        mat = scipy.io.loadmat(path)

        if 'inputfeats_combined' not in mat:
            raise KeyError(f"'inputfeats_combined' not found in {f}. Keys: {mat.keys()}")

        data = mat['inputfeats_combined']
        data = np.transpose(data, (3, 0, 1, 2))  # (38, 16, 109, 1) or similar
        all_data.append(data)
        all_filenames.append(f)

    stacked = np.concatenate(all_data, axis=0)
    return stacked, all_filenames

def compute_azimuth_rmse(preds, filelist, label_df, snr_val):
    rows = []
    for i, fname in enumerate(filelist):
        block_preds = preds[i * 38: (i + 1) * 38]
        pred_mean = np.mean(block_preds, axis=0)

        rows.append({
            'FileName': os.path.basename(fname),
            'Pred_x': pred_mean[0],
            'Pred_y': pred_mean[1],
            'Pred_z': pred_mean[2]
        })

    result_df = pd.DataFrame(rows)
    result_df['FileName'] = result_df['FileName'].apply(os.path.basename)
    label_df['FileName'] = label_df['FileName'].apply(os.path.basename)

    merged_df = pd.merge(result_df, label_df, on='FileName', how='left')
    
    merged_df[['FileName', 'Pred_x', 'Pred_y', 'Pred_z', 'True_x', 'True_y', 'True_z']].to_csv(
        f"predictions_SNR_{snr_val}.csv", index=False)

    pred_xy = merged_df[['Pred_x', 'Pred_y']].values
    true_xy = merged_df[['True_x', 'True_y']].values

    pred_unit = pred_xy / np.linalg.norm(pred_xy, axis=1, keepdims=True)
    true_unit = true_xy / np.linalg.norm(true_xy, axis=1, keepdims=True)

    dot_products = np.sum(pred_unit * true_unit, axis=1)
    clipped_dots = np.clip(dot_products, -1.0, 1.0)
    angular_errors = np.arccos(clipped_dots) * (180.0 / np.pi)

    rmse = np.sqrt(np.mean(angular_errors ** 2))
    return rmse

# === MAIN SCRIPT ===
def main():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            'NSASMetric': NSASMetric,
            'AngularErrorMetric': AngularErrorMetric
        }
    )
    print("✅ Model loaded.")

    # === Load labels and convert spherical -> Cartesian ===
    label_df = pd.read_csv(LABEL_CSV)
    label_df = label_df.rename(columns={"filename": "FileName", "theta": "Theta", "phi": "Phi"})

    theta_rad = np.deg2rad(label_df["Theta"].values)  # elevation
    phi_rad   = np.deg2rad(label_df["Phi"].values)    # azimuth

    label_df["True_x"] = np.cos(theta_rad) * np.cos(phi_rad)
    label_df["True_y"] = np.cos(theta_rad) * np.sin(phi_rad)
    label_df["True_z"] = np.sin(theta_rad)
    label_df = label_df[["FileName", "True_x", "True_y", "True_z"]]

    results = []

    for snr in SNR_LEVELS:
        test_dir = os.path.join(BASE_SNR_DIR, f"SNR{snr}")
        print(f"\n📁 Processing SNR {snr} dB...")

        x_test, filelist = load_and_stack_test_data(test_dir)
        print(f"📦 Test data shape: {x_test.shape}")

        # Match ground truth with each frame
        y_true = []
        for fname in filelist:
            row = label_df[label_df["FileName"] == os.path.basename(fname)].iloc[0]
            coords = np.array([row["True_x"], row["True_y"], row["True_z"]])
            y_true.append(np.tile(coords, (38, 1)))
        y_true = np.vstack(y_true)

        # Evaluate → get metrics
        eval_results = model.evaluate(x_test, y_true, verbose=0, return_dict=True)
        loss = eval_results["loss"]
        angular_error = eval_results.get("angular_error", np.nan)
        nsas = eval_results.get("nsas", np.nan)

        # Predictions → RMSE
        preds = model.predict(x_test, batch_size=32, verbose=0)
        rmse = compute_azimuth_rmse(preds, filelist, label_df.copy(), snr)

        results.append({
            "SNR (dB)": snr,
            "Loss": loss,
            "RMSE": rmse,
            "AngularError": angular_error,
            "NSAS": nsas
        })

        print(f"✅ SNR {snr} dB: Loss={loss:.4f}, RMSE={rmse:.2f}, AngularError={angular_error:.2f}, NSAS={nsas:.4f}")

    # === Save results ===
    results_df = pd.DataFrame(results)
    results_df.to_excel("CNN_SNR.xlsx", index=False)
    print("\n📊 Results saved to CNN_SNR.xlsx")
    print(results_df)

    # === Plot each metric vs SNR ===
    plt.figure(figsize=(10, 6))
    for metric in ["Loss", "RMSE", "AngularError", "NSAS"]:
        plt.plot(results_df["SNR (dB)"], results_df[metric], marker='o', label=metric)

    plt.title("CNN Performance vs SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Metric Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cnn_metrics_vs_snr.png")
    plt.show()
    print("\n📊 Plot saved as cnn_metrics_vs_snr.png")

if __name__ == "__main__":
    main()
