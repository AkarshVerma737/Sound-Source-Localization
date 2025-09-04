# import os
# import numpy as np
# import scipy.io
# import tensorflow as tf
# import pandas as pd
# import matplotlib.pyplot as plt

# # === CONFIGURATION ===
# MODEL_PATH = 'D:/Projects/Learning based Beamforging/SH/Output/Transformer/best_by_val_angle.keras'
# BASE_SNR_DIR = 'D:/Projects/Learning based Beamforging/SH/Generated Audio/TESTDATA'
# LABEL_CSV = 'D:/Projects/Learning based Beamforging/SH/Generated Audio/TESTDATA/test_labels.csv'
# SNR_LEVELS = [5, 8, 11, 14, 17, 20]  # SNRs in dB

# # === CUSTOM METRICS ===
# class NSASMetric(tf.keras.metrics.Metric):
#     def __init__(self, d_max=1.0, name='nsas', **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.d_max = tf.constant(d_max, dtype=tf.float32)
#         self.total = self.add_weight(name='total', initializer='zeros')
#         self.count = self.add_weight(name='count', initializer='zeros')

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         true_dir = tf.math.l2_normalize(y_true, axis=-1)
#         pred_dir = tf.math.l2_normalize(y_pred, axis=-1)
#         dot_product = tf.reduce_sum(true_dir * pred_dir, axis=-1)
#         angular_error = tf.acos(tf.clip_by_value(dot_product, -1.0, 1.0)) * (180.0 / np.pi)
#         angular_score = 1 - (angular_error / 180.0)
#         distance_error = tf.norm(y_true - y_pred, axis=-1)
#         distance_score = 1 - (distance_error / self.d_max)
#         nsas_values = angular_score * distance_score
#         self.total.assign_add(tf.reduce_sum(nsas_values))
#         self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

#     def result(self):
#         return self.total / self.count

#     def reset_state(self):
#         self.total.assign(0.0)
#         self.count.assign(0.0)

# class AngularErrorMetric(tf.keras.metrics.Metric):
#     def __init__(self, name='angular_error', **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.total = self.add_weight(name='total', initializer='zeros')
#         self.count = self.add_weight(name='count', initializer='zeros')

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         true_dir = tf.math.l2_normalize(y_true, axis=-1)
#         pred_dir = tf.math.l2_normalize(y_pred, axis=-1)
#         dot_product = tf.reduce_sum(true_dir * pred_dir, axis=-1)
#         angular_error = tf.acos(tf.clip_by_value(dot_product, -1.0, 1.0)) * (180.0 / np.pi)
#         self.total.assign_add(tf.reduce_sum(angular_error))
#         self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

#     def result(self):
#         return self.total / self.count

#     def reset_state(self):
#         self.total.assign(0.0)
#         self.count.assign(0.0)

# # === LOAD & STACK TEST DATA ===
# def load_and_stack_test_data(test_dir):
#     mat_files = [f for f in os.listdir(test_dir) if f.endswith('.mat')]
#     all_data = []
#     all_filenames = []

#     for f in sorted(mat_files):
#         path = os.path.join(test_dir, f)
#         mat = scipy.io.loadmat(path)

#         if 'inputfeats_combined' not in mat:
#             raise KeyError(f"'inputfeats_combined' not found in {f}. Keys: {mat.keys()}")

#         features = mat['inputfeats_combined']  # (16, 109, 2, 38)
#         features = np.transpose(features, (3, 0, 1, 2))  # → (38, 16, 109, 2)
#         all_data.append(features)
#         all_filenames.append(f)

#     stacked = np.stack(all_data, axis=0)  # (N, 38, 16, 109, 2)
#     return stacked.astype(np.float32), all_filenames

# # === COMPUTE AZIMUTH RMSE ===
# def compute_azimuth_rmse(preds, filelist, label_df, snr_level):
#     rows = []
#     for i, fname in enumerate(filelist):
#         pred_x, pred_y, pred_z = preds[i]
#         pred_phi = np.degrees(np.arctan2(pred_y, pred_x)) % 360  # azimuth in degrees [0, 360)

#         rows.append({
#             'FileName': os.path.basename(fname),
#             'Pred_x': pred_x,
#             'Pred_y': pred_y,
#             'Pred_z': pred_z,
#             'Pred_phi': pred_phi
#         })

#     result_df = pd.DataFrame(rows)
#     result_df['FileName'] = result_df['FileName'].apply(os.path.basename)
#     label_df['FileName'] = label_df['FileName'].apply(os.path.basename)

#     merged_df = pd.merge(result_df, label_df, on='FileName', how='left')

#     # Save predictions
#     merged_df[['FileName', 'Pred_x', 'Pred_y', 'Pred_z', 'Pred_phi', 'theta', 'phi']].to_csv(
#         f"predictions_SNR_{snr_level}.csv", index=False)

#     # Compute azimuth RMSE (degrees)
#     true_phi = merged_df['phi'].values % 360
#     pred_phi = merged_df['Pred_phi'].values % 360
#     angular_errors = np.abs(pred_phi - true_phi)
#     angular_errors = np.minimum(angular_errors, 360 - angular_errors)
#     rmse = np.sqrt(np.mean(angular_errors ** 2))
#     return rmse

# # === MAIN ===
# def main():
#     # Load model
#     model = tf.keras.models.load_model(
#         MODEL_PATH,
#         custom_objects={
#             'NSASMetric': NSASMetric,
#             'AngularErrorMetric': AngularErrorMetric
#         }
#     )
#     print("✅ Model loaded.")

#     # Load ground truth labels
#     label_df = pd.read_csv(LABEL_CSV, usecols=["FileName", "theta", "phi"])
#     label_df['FileName'] = label_df['FileName'].apply(os.path.basename)

#     snr_errors = []

#     for snr in SNR_LEVELS:
#         test_dir = os.path.join(BASE_SNR_DIR, f"SNR{snr}")
#         print(f"\n📁 Processing SNR {snr} dB...")

#         x_test, filelist = load_and_stack_test_data(test_dir)
#         print(f"📦 Test data shape: {x_test.shape}")

#         preds = model.predict(x_test, batch_size=32, verbose=0)
#         print(f"✅ Predictions done for SNR {snr}.")

#         rmse = compute_azimuth_rmse(preds, filelist, label_df.copy(), snr)
#         snr_errors.append((snr, rmse))
#         print(f"📐 Azimuth RMSE @ {snr} dB = {rmse:.2f}°")

#     # === PLOT RMSE vs SNR ===
#     snrs, errors = zip(*snr_errors)
#     plt.figure(figsize=(8, 5))
#     plt.ylim(0, 20)
#     plt.plot(snrs, errors, marker='o', linewidth=2, color='blue')
#     plt.title("Azimuth RMSE vs SNR")
#     plt.xlabel("SNR (dB)")
#     plt.ylabel("Azimuth RMSE (degrees)")
#     plt.grid(True)
#     plt.xticks(snrs)
#     plt.tight_layout()
#     plt.savefig("azimuth_rmse_vs_snr.png")
#     plt.show()
#     print("\n📊 Plot saved as azimuth_rmse_vs_snr2.png")

# if __name__ == "__main__":
#     main()

import os
import numpy as np
import scipy.io
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import time

# === CONFIGURATION ===
MODEL_PATH = 'D:/Projects/Learning based Beamforging/SH/Output/Transformer/best_by_val_angle.keras'
BASE_SNR_DIR = 'D:/Projects/Learning based Beamforging/SH/Generated Audio/TESTDATA'
LABEL_CSV = 'D:/Projects/Learning based Beamforging/SH/Generated Audio/TESTDATA/test_labels.csv'
SNR_LEVELS = [5, 8, 11, 14, 17, 20]  # SNRs in dB
OUTPUT_EXCEL = "TransformerSNR.xlsx"

# === CUSTOM METRICS (same names as in your model) ===
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
        distance_error = tf.norm(y_true - y_pred, axis=-1)    # uses raw vectors, matching your class
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

# === HELPERS ===
def sph2cart(theta_deg, phi_deg):
    """
    Convert spherical (theta = elevation in degrees, phi = azimuth in degrees)
    to Cartesian unit vector (x,y,z). Assumes radius=1.
    """
    th = np.deg2rad(theta_deg)
    ph = np.deg2rad(phi_deg)
    x = np.cos(th) * np.cos(ph)
    y = np.cos(th) * np.sin(ph)
    z = np.sin(th)
    return np.vstack([x, y, z]).T.astype(np.float32)

def load_and_stack_test_data(test_dir):
    mat_files = [f for f in os.listdir(test_dir) if f.endswith('.mat')]
    all_data, all_filenames = [], []
    for f in sorted(mat_files):
        path = os.path.join(test_dir, f)
        mat = scipy.io.loadmat(path)
        if 'inputfeats_combined' not in mat:
            raise KeyError(f"'inputfeats_combined' not found in {f}. Keys: {list(mat.keys())}")
        features = mat['inputfeats_combined']  # (16, 109, 2, 38)
        features = np.transpose(features, (3, 0, 1, 2))  # → (38, 16, 109, 2)
        all_data.append(features)
        all_filenames.append(f)
    stacked = np.stack(all_data, axis=0)  # (N, 38, 16, 109, 2)
    return stacked.astype(np.float32), all_filenames

def build_y_true_from_labels(filelist, label_df):
    """
    Create y_true (Nx3) aligned with predictions order.
    """
    order_df = pd.DataFrame({'FileName': [os.path.basename(f) for f in filelist]})
    label_df = label_df.copy()
    label_df['FileName'] = label_df['FileName'].apply(os.path.basename)
    merged = order_df.merge(label_df[['FileName', 'theta', 'phi']], on='FileName', how='left')

    if merged[['theta', 'phi']].isna().any().any():
        missing = merged[merged[['theta', 'phi']].isna().any(axis=1)]['FileName'].tolist()
        raise ValueError(f"Missing labels for files: {missing[:10]} ...")

    y_true = sph2cart(merged['theta'].to_numpy(), merged['phi'].to_numpy())
    return y_true, merged

def compute_azimuth_rmse(preds, filelist, label_df, snr_level):
    """
    RMSE of azimuth (degrees) between predicted phi (from pred x,y) and label phi.
    """
    rows = []
    for i, fname in enumerate(filelist):
        pred_x, pred_y, pred_z = preds[i]
        pred_phi = np.degrees(np.arctan2(pred_y, pred_x)) % 360
        rows.append({'FileName': os.path.basename(fname),
                     'Pred_x': pred_x, 'Pred_y': pred_y, 'Pred_z': pred_z, 'Pred_phi': pred_phi})

    result_df = pd.DataFrame(rows)
    result_df['FileName'] = result_df['FileName'].apply(os.path.basename)

    label_df = label_df.copy()
    label_df['FileName'] = label_df['FileName'].apply(os.path.basename)
    merged_df = pd.merge(result_df, label_df, on='FileName', how='left')

    # Save per-SNR predictions (optional but handy)
    merged_df[['FileName', 'Pred_x', 'Pred_y', 'Pred_z', 'Pred_phi', 'theta', 'phi']].to_csv(
        f"predictions_SNR_{snr_level}.csv", index=False)

    true_phi = merged_df['phi'].values % 360
    pred_phi = merged_df['Pred_phi'].values % 360
    angular_errors = np.abs(pred_phi - true_phi)
    angular_errors = np.minimum(angular_errors, 360 - angular_errors)
    rmse = float(np.sqrt(np.mean(angular_errors ** 2)))
    return rmse

# === MAIN ===
def main():
    # Load model with custom metrics registered
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'NSASMetric': NSASMetric, 'AngularErrorMetric': AngularErrorMetric}
    )
    print("✅ Model loaded.")

    # Load labels
    label_df = pd.read_csv(LABEL_CSV, usecols=["FileName", "theta", "phi"])
    label_df['FileName'] = label_df['FileName'].apply(os.path.basename)

    results = []

    for snr in SNR_LEVELS:
        test_dir = os.path.join(BASE_SNR_DIR, f"SNR{snr}")
        print(f"\n📁 Processing SNR {snr} dB...")

        x_test, filelist = load_and_stack_test_data(test_dir)
        print(f"📦 Test data shape: {x_test.shape}")


        # Build ground-truth vectors aligned to file order (CRITICAL FIX)
        y_true, ordered_labels = build_y_true_from_labels(filelist, label_df)

        # Evaluate model to get Loss + model metrics (AngularErrorMetric, NSASMetric)
        eval_results = model.evaluate(x_test, y_true, batch_size=32, verbose=0, return_dict=True)

        # Predict for azimuth RMSE
        preds = model.predict(x_test, batch_size=32, verbose=0)
        print(f"✅ Predictions done for SNR {snr}.")

        rmse = compute_azimuth_rmse(preds, filelist, ordered_labels[['FileName', 'theta', 'phi']], snr)

        results.append({
            "SNR (dB)": snr,
            "Loss": float(eval_results.get("loss", np.nan)),
            "Angular Error (°)": float(eval_results.get("angular_error", np.nan)),
            "NSAS": float(eval_results.get("nsas", np.nan)),
            "Azimuth RMSE (°)": rmse
        })

        print(f"📊 @ {snr} dB → Loss={results[-1]['Loss']:.6f}, "
              f"AngErr={results[-1]['Angular Error (°)']:.3f}°, "
              f"NSAS={results[-1]['NSAS']:.6f}, RMSE={rmse:.3f}°")

    # Results table → Excel
    results_df = pd.DataFrame(results).sort_values("SNR (dB)")
    results_df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\n✅ Results saved to {OUTPUT_EXCEL}")

    # === PLOTS ===
    # Keep separate plots for each metric vs SNR
    metrics_and_labels = [
        ("Loss", "Loss"),
        ("Angular Error (°)", "Angular Error (°)"),
        ("NSAS", "NSAS"),
        ("Azimuth RMSE (°)", "Azimuth RMSE (°)"),
    ]

    for col, ylabel in metrics_and_labels:
        plt.figure(figsize=(8, 5))
        plt.plot(results_df["SNR (dB)"], results_df[col], marker='o', linewidth=2)
        plt.title(f"{ylabel} vs SNR")
        plt.xlabel("SNR (dB)")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.xticks(results_df["SNR (dB)"])
        fname = f"{col.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('°','deg')}_vs_snr.png"
        plt.tight_layout()
        plt.savefig(fname)
        plt.show()
        print(f"🖼️ Saved plot: {fname}")

if __name__ == "__main__":
    main()
