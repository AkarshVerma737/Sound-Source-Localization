# # import os
# # import numpy as np
# # import scipy.io
# # import tensorflow as tf
# # import pandas as pd
# # import matplotlib.pyplot as plt

# # # === CONFIGURATION ===
# # MODEL_PATH = 'D:/Projects/Learning based Beamforging/SH/Output/Regression/cartesian_batch10/best_model.44-0.004050.h5'
# # BASE_BETA_DIR = 'D:/Projects/Learning based Beamforging/SH/Generated Audio/TESTDATAREV'  # <-- ✅ New base folder
# # LABEL_CSV = 'D:/Projects/Learning based Beamforging/SH/Generated Audio/TESTDATAREV/test_labels.csv'
# # BETA_VALUES = [round(x, 1) for x in np.arange(0.2, 1.1, 0.2)] 

# # # === CUSTOM METRICS (REQUIRED TO LOAD MODEL) ===
# # class NSASMetric(tf.keras.metrics.Metric):
# #     def __init__(self, d_max=1.0, name='nsas', **kwargs):
# #         super().__init__(name=name, **kwargs)
# #         self.d_max = tf.constant(d_max, dtype=tf.float32)
# #         self.total = self.add_weight(name='total', initializer='zeros')
# #         self.count = self.add_weight(name='count', initializer='zeros')

# #     def update_state(self, y_true, y_pred, sample_weight=None):
# #         true_dir = tf.math.l2_normalize(y_true, axis=-1)
# #         pred_dir = tf.math.l2_normalize(y_pred, axis=-1)
# #         dot_product = tf.reduce_sum(true_dir * pred_dir, axis=-1)
# #         angular_error = tf.acos(tf.clip_by_value(dot_product, -1.0, 1.0)) * (180.0 / np.pi)
# #         angular_score = 1 - (angular_error / 180.0)

# #         distance_error = tf.norm(y_true - y_pred, axis=-1)
# #         distance_score = 1 - (distance_error / self.d_max)

# #         nsas_values = angular_score * distance_score
# #         self.total.assign_add(tf.reduce_sum(nsas_values))
# #         self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

# #     def result(self):
# #         return self.total / self.count

# #     def reset_state(self):
# #         self.total.assign(0.0)
# #         self.count.assign(0.0)

# # class AngularErrorMetric(tf.keras.metrics.Metric):
# #     def __init__(self, name='angular_error', **kwargs):
# #         super().__init__(name=name, **kwargs)
# #         self.total = self.add_weight(name='total', initializer='zeros')
# #         self.count = self.add_weight(name='count', initializer='zeros')

# #     def update_state(self, y_true, y_pred, sample_weight=None):
# #         true_dir = tf.math.l2_normalize(y_true, axis=-1)
# #         pred_dir = tf.math.l2_normalize(y_pred, axis=-1)
# #         dot_product = tf.reduce_sum(true_dir * pred_dir, axis=-1)
# #         angular_error = tf.acos(tf.clip_by_value(dot_product, -1.0, 1.0)) * (180.0 / np.pi)
# #         self.total.assign_add(tf.reduce_sum(angular_error))
# #         self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

# #     def result(self):
# #         return self.total / self.count

# #     def reset_state(self):
# #         self.total.assign(0.0)
# #         self.count.assign(0.0)

# # # === UTILS ===
# # def load_and_stack_test_data(test_dir):
# #     mat_files = [f for f in os.listdir(test_dir) if f.endswith('.mat')]
# #     all_data = []
# #     all_filenames = []

# #     for f in mat_files:
# #         path = os.path.join(test_dir, f)
# #         mat = scipy.io.loadmat(path)

# #         if 'inputfeats_combined' not in mat:
# #             raise KeyError(f"'inputfeats_combined' not found in {f}. Keys: {mat.keys()}")

# #         data = mat['inputfeats_combined']
# #         data = np.transpose(data, (3, 0, 1, 2))  # (38, 16, 109, 1)
# #         all_data.append(data)
# #         all_filenames.append(f)

# #     stacked = np.concatenate(all_data, axis=0)
# #     return stacked, all_filenames

# # def compute_azimuth_rmse(preds, filelist, label_df, beta_val):
# #     rows = []
# #     for i, fname in enumerate(filelist):
# #         block_preds = preds[i * 38: (i + 1) * 38]
# #         pred_mean = np.mean(block_preds, axis=0)

# #         rows.append({
# #             'FileName': os.path.basename(fname),
# #             'Pred_x': pred_mean[0],
# #             'Pred_y': pred_mean[1],
# #             'Pred_z': pred_mean[2]
# #         })

# #     result_df = pd.DataFrame(rows)
# #     result_df['FileName'] = result_df['FileName'].apply(os.path.basename)
# #     label_df['FileName'] = label_df['FileName'].apply(os.path.basename)

# #     merged_df = pd.merge(result_df, label_df, on='FileName', how='left')
    
# #     # ✅ Save predictions
# #     merged_df[['FileName', 'Pred_x', 'Pred_y', 'Pred_z', 'True_x', 'True_y', 'True_z']].to_csv(
# #         f"predictions_RT60_{int(beta_val * 10):02d}.csv", index=False)

# #     pred_xy = merged_df[['Pred_x', 'Pred_y']].values
# #     true_xy = merged_df[['True_x', 'True_y']].values

# #     pred_unit = pred_xy / np.linalg.norm(pred_xy, axis=1, keepdims=True)
# #     true_unit = true_xy / np.linalg.norm(true_xy, axis=1, keepdims=True)

# #     dot_products = np.sum(pred_unit * true_unit, axis=1)
# #     clipped_dots = np.clip(dot_products, -1.0, 1.0)
# #     angular_errors = np.arccos(clipped_dots) * (180.0 / np.pi)

# #     rmse = np.sqrt(np.mean(angular_errors ** 2))
# #     return rmse

# # # === MAIN SCRIPT ===
# # def main():
# #     model = tf.keras.models.load_model(
# #         MODEL_PATH,
# #         custom_objects={
# #             'NSASMetric': NSASMetric,
# #             'AngularErrorMetric': AngularErrorMetric
# #         }
# #     )
# #     print("✅ Model loaded.")

# #     label_df = pd.read_csv(LABEL_CSV, usecols=["FileName", "x", "y", "z"])
# #     label_df = label_df.rename(columns={"x": "True_x", "y": "True_y", "z": "True_z"})

# #     beta_errors = []

# #     for beta in BETA_VALUES:
# #         test_dir = os.path.join(BASE_BETA_DIR, f"REV{beta}")
# #         print(f"\n📁 Processing RT60 = {beta} (folder: {beta})")

# #         x_test, filelist = load_and_stack_test_data(test_dir)
# #         print(f"📦 Test data shape: {x_test.shape}")

# #         preds = model.predict(x_test, batch_size=32, verbose=0)
# #         print(f"✅ Predictions done for RT60 {beta}.")

# #         rmse = compute_azimuth_rmse(preds, filelist, label_df.copy(), beta)
# #         beta_errors.append((beta, rmse))
# #         print(f"📐 Azimuth RMSE @ RT60 {beta} = {rmse:.2f}°")

# #     # === PLOT RMSE vs RT60 ===
# #     betas, errors = zip(*beta_errors)
# #     plt.figure(figsize=(8, 5))
# #     plt.plot(betas, errors, marker='o', linewidth=2, color='green')
# #     plt.title("Azimuth RMSE vs RT60")
# #     plt.xlabel("RT60 Beta Value")
# #     plt.ylabel("Azimuth RMSE (degrees)")
# #     plt.ylim(0, 25)
# #     plt.grid(True)
# #     plt.xticks(betas)
# #     plt.tight_layout()
# #     plt.savefig("azimuth_rmse_vs_rt60.png")
# #     plt.show()
# #     print("\n📊 Plot saved as azimuth_rmse_vs_rt60.png")

# # if __name__ == "__main__":
# #     main()

# import os
# import numpy as np
# import scipy.io
# import tensorflow as tf
# import pandas as pd
# import matplotlib.pyplot as plt

# # === CONFIGURATION ===
# MODEL_PATH = 'D:/Projects/Learning based Beamforging/SH/Output/Regression/cartesian_batch10/best_model.44-0.004050.h5'
# BASE_BETA_DIR = 'D:/Projects/Learning based Beamforging/SH/Generated Audio/TESTDATAREV'  # <-- ✅ New base folder
# LABEL_CSV = 'D:/Projects/Learning based Beamforging/SH/Generated Audio/TESTDATAREV/test_labels.csv'
# BETA_VALUES = [round(x, 1) for x in np.arange(0.2, 1.1, 0.2)] 

# # === CUSTOM METRICS (REQUIRED TO LOAD MODEL) ===
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

# # === UTILS ===
# def load_and_stack_test_data(test_dir):
#     mat_files = [f for f in os.listdir(test_dir) if f.endswith('.mat')]
#     all_data = []
#     all_filenames = []

#     for f in mat_files:
#         path = os.path.join(test_dir, f)
#         mat = scipy.io.loadmat(path)

#         if 'inputfeats_combined' not in mat:
#             raise KeyError(f"'inputfeats_combined' not found in {f}. Keys: {mat.keys()}")

#         data = mat['inputfeats_combined']
#         data = np.transpose(data, (3, 0, 1, 2))  # (38, 16, 109, 1)
#         all_data.append(data)
#         all_filenames.append(f)

#     stacked = np.concatenate(all_data, axis=0)
#     return stacked, all_filenames

# def compute_azimuth_rmse(preds, filelist, label_df, beta_val):
#     rows = []
#     for i, fname in enumerate(filelist):
#         block_preds = preds[i * 38: (i + 1) * 38]
#         pred_mean = np.mean(block_preds, axis=0)

#         rows.append({
#             'FileName': os.path.basename(fname),
#             'Pred_x': pred_mean[0],
#             'Pred_y': pred_mean[1],
#             'Pred_z': pred_mean[2]
#         })

#     result_df = pd.DataFrame(rows)
#     result_df['FileName'] = result_df['FileName'].apply(os.path.basename)
#     label_df['FileName'] = label_df['FileName'].apply(os.path.basename)

#     merged_df = pd.merge(result_df, label_df, on='FileName', how='left')
    
#     # ✅ Save predictions
#     merged_df[['FileName', 'Pred_x', 'Pred_y', 'Pred_z', 'True_x', 'True_y', 'True_z']].to_csv(
#         f"predictions_RT60_{int(beta_val * 10):02d}.csv", index=False)

#     pred_xy = merged_df[['Pred_x', 'Pred_y']].values
#     true_xy = merged_df[['True_x', 'True_y']].values

#     pred_unit = pred_xy / np.linalg.norm(pred_xy, axis=1, keepdims=True)
#     true_unit = true_xy / np.linalg.norm(true_xy, axis=1, keepdims=True)

#     dot_products = np.sum(pred_unit * true_unit, axis=1)
#     clipped_dots = np.clip(dot_products, -1.0, 1.0)
#     angular_errors = np.arccos(clipped_dots) * (180.0 / np.pi)

#     rmse = np.sqrt(np.mean(angular_errors ** 2))
#     return rmse

# # === MAIN SCRIPT ===
# def main():
#     model = tf.keras.models.load_model(
#         MODEL_PATH,
#         custom_objects={
#             'NSASMetric': NSASMetric,
#             'AngularErrorMetric': AngularErrorMetric
#         }
#     )
#     print("✅ Model loaded.")

#     # === Load labels and convert spherical -> Cartesian ===
#     label_df = pd.read_csv(LABEL_CSV)

#     # Standardize column names
#     label_df = label_df.rename(columns={"filename": "FileName", "theta": "Theta", "phi": "Phi"})

#     # Convert to radians
#     theta_rad = np.deg2rad(label_df["Theta"].values)  # elevation
#     phi_rad   = np.deg2rad(label_df["Phi"].values)    # azimuth

#     # r = 1, spherical → Cartesian
#     label_df["True_x"] = np.cos(theta_rad) * np.cos(phi_rad)
#     label_df["True_y"] = np.cos(theta_rad) * np.sin(phi_rad)
#     label_df["True_z"] = np.sin(theta_rad)

#     # Keep only required columns
#     label_df = label_df[["FileName", "True_x", "True_y", "True_z"]]

#     beta_errors = []

#     for beta in BETA_VALUES:
#         test_dir = os.path.join(BASE_BETA_DIR, f"REV{beta}")
#         print(f"\n📁 Processing RT60 = {beta} (folder: {beta})")

#         x_test, filelist = load_and_stack_test_data(test_dir)
#         print(f"📦 Test data shape: {x_test.shape}")

#         preds = model.predict(x_test, batch_size=32, verbose=0)
#         print(f"✅ Predictions done for RT60 {beta}.")

#         rmse = compute_azimuth_rmse(preds, filelist, label_df.copy(), beta)
#         beta_errors.append((beta, rmse))
#         print(f"📐 Azimuth RMSE @ RT60 {beta} = {rmse:.2f}°")

#     # === PLOT RMSE vs RT60 ===
#     betas, errors = zip(*beta_errors)
#     plt.figure(figsize=(8, 5))
#     plt.plot(betas, errors, marker='o', linewidth=2, color='green')
#     plt.title("Azimuth RMSE vs RT60")
#     plt.xlabel("RT60 Beta Value")
#     plt.ylabel("Azimuth RMSE (degrees)")
#     plt.ylim(0, 25)
#     plt.grid(True)
#     plt.xticks(betas)
#     plt.tight_layout()
#     plt.savefig("azimuth_rmse_vs_rt60.png")
#     plt.show()
#     print("\n📊 Plot saved as azimuth_rmse_vs_rt60.png")

# if __name__ == "__main__":
#     main()
import os
import numpy as np
import scipy.io
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIGURATION ===
MODEL_PATH = 'D:/Projects/Learning based Beamforging/SH/Output/Regression/cartesian_batch10/best_model.44-0.004050.h5'
BASE_BETA_DIR = 'D:/Projects/Learning based Beamforging/SH/Generated Audio/TESTDATAREV'
LABEL_CSV = 'D:/Projects/Learning based Beamforging/SH/Generated Audio/TESTDATAREV/test_labels.csv'
BETA_VALUES = [round(x, 1) for x in np.arange(0.2, 1.1, 0.2)] 

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
        data = np.transpose(data, (3, 0, 1, 2))  # (38, 16, 109, 1)
        all_data.append(data)
        all_filenames.append(f)

    stacked = np.concatenate(all_data, axis=0)
    return stacked, all_filenames

def compute_azimuth_rmse(preds, filelist, label_df, beta_val):
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
    
    # Save predictions for reference
    merged_df[['FileName', 'Pred_x', 'Pred_y', 'Pred_z', 'True_x', 'True_y', 'True_z']].to_csv(
        f"predictions_RT60_{int(beta_val * 10):02d}.csv", index=False)

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

    for beta in BETA_VALUES:
        test_dir = os.path.join(BASE_BETA_DIR, f"REV{beta}")
        print(f"\n📁 Processing RT60 = {beta}")

        x_test, filelist = load_and_stack_test_data(test_dir)
        print(f"📦 Test data shape: {x_test.shape}")

        # Match ground truth with each frame
        y_true = []
        for fname in filelist:
            row = label_df[label_df["FileName"] == os.path.basename(fname)].iloc[0]
            coords = np.array([row["True_x"], row["True_y"], row["True_z"]])
            y_true.append(np.tile(coords, (38, 1)))  # repeat for 38 frames
        y_true = np.vstack(y_true)

        # Evaluate on this subset
        # Evaluate on this subset (returns dict of all metrics)
        eval_results = model.evaluate(x_test, y_true, verbose=0, return_dict=True)

        loss = eval_results["loss"]
        angular_error = eval_results.get("angular_error", np.nan)
        nsas = eval_results.get("nsas", np.nan)

        
        # Predictions
        preds = model.predict(x_test, batch_size=32, verbose=0)
        rmse = compute_azimuth_rmse(preds, filelist, label_df.copy(), beta)

        results.append({
            "RT60": beta,
            "Loss": loss,
            "RMSE": rmse,
            "AngularError": angular_error,
            "NSAS": nsas
        })

        print(f"✅ RT60 {beta}: Loss={loss:.4f}, RMSE={rmse:.2f}, AngularError={angular_error:.2f}, NSAS={nsas:.4f}")

    # === Save results table ===
    results_df = pd.DataFrame(results)
    results_df.to_excel("CNNRT60.xlsx", index=False)
    print("\n📊 Results saved to CNNRT60.xlsx")
    print(results_df)

if __name__ == "__main__":
    main()
