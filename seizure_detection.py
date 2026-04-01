
import sys

import os
import numpy as np
import pandas as pd
import mne
import warnings
from scipy import signal, stats
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, recall_score)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')


# CONFIGURATION — change paths here

DATA_DIR = "data/chb01"           
SUMMARY_FILE = "data/chb01/chb01-summary.txt"
WINDOW_SEC = 5                    
SAMPLE_RATE = 256               
WINDOW_SAMPLES = WINDOW_SEC * SAMPLE_RATE  
N_FEATURES_PER_CHANNEL = 32      
SEIZURE_THRESHOLD = 0.60          
N_TREES = 50                      
EVENT_GAP_SEC = 10                
OVERLAP_THRESHOLD = 0.70          


# PART 1: PARSE SUMMARY FILE

def parse_summary(summary_path):
    seizure_times = {}
    current_file = None

    with open(summary_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("File Name:"):
            current_file = line.split(": ")[1].strip()
            seizure_times[current_file] = []

        elif line.startswith("Seizure") and "Start Time" in line:
            start = int(line.split(": ")[1].replace("seconds", "").strip())
            i += 1
            end_line = lines[i].strip()
            end = int(end_line.split(": ")[1].replace("seconds", "").strip())
            seizure_times[current_file].append((start, end))

        i += 1

    return seizure_times


# ============================================================
# PART 2: FEATURE EXTRACTION
# Extracts 23 features from a single 5-second channel segment
# (paper uses 92 total; this covers the main categories)
# ============================================================
def extract_features(segment, fs=256):
    features = {}

    # --- TIME DOMAIN STATISTICAL ---
    features['mean'] = np.mean(segment)
    features['std'] = np.std(segment)
    features['var'] = np.var(segment)
    features['skewness'] = skew(segment)
    features['kurtosis'] = kurtosis(segment)
    features['rms'] = np.sqrt(np.mean(segment**2))
    features['peak_to_peak'] = np.ptp(segment)
    features['zero_crossing_rate'] = ((np.diff(np.sign(segment)) != 0).sum()
                                      / len(segment))

    # --- FREQUENCY DOMAIN ---
    freqs, psd = signal.welch(segment, fs=fs, nperseg=min(256, len(segment)))

    def band_power(freqs, psd, low, high):
        idx = np.logical_and(freqs >= low, freqs <= high)
        return np.trapezoid(psd[idx], freqs[idx])

    features['delta_power'] = band_power(freqs, psd, 0.5, 4)    # 0.5–4 Hz
    features['theta_power'] = band_power(freqs, psd, 4, 8)      # 4–8 Hz
    features['alpha_power'] = band_power(freqs, psd, 8, 13)     # 8–13 Hz
    features['beta_power']  = band_power(freqs, psd, 13, 30)    # 13–30 Hz
    features['gamma_power'] = band_power(freqs, psd, 30, 70)    # 30–70 Hz
    features['total_power'] = band_power(freqs, psd, 0.5, 70)

    # Spectral ratios
    total = features['total_power'] + 1e-10
    features['delta_ratio'] = features['delta_power'] / total
    features['theta_ratio'] = features['theta_power'] / total
    features['alpha_ratio'] = features['alpha_power'] / total
    features['beta_ratio']  = features['beta_power']  / total
    features['gamma_ratio'] = features['gamma_power'] / total

    # Dominant frequency
    features['dominant_freq'] = freqs[np.argmax(psd)]
    features['spectral_entropy'] = stats.entropy(psd + 1e-10)

    # --- ENTROPY ---
    # Sample Entropy (simplified)
    def sample_entropy(x, m=2, r_factor=0.2):
        r = r_factor * np.std(x)
        N = len(x)
        def count_matches(template_length):
            count = 0
            for i in range(N - template_length):
                template = x[i:i+template_length]
                for j in range(i+1, N - template_length):
                    if np.max(np.abs(x[j:j+template_length] - template)) < r:
                        count += 1
            return count
        # Fast approximation using only first 200 samples
        x_short = x[:200]
        N = len(x_short)
        B = count_matches(m) + 1e-10
        A = count_matches(m + 1) + 1e-10
        return -np.log(A / B)

    try:
        features['sample_entropy'] = sample_entropy(segment[:200])
    except Exception:
        features['sample_entropy'] = 0.0

    return list(features.values())

# PART 3: LOAD EDF + SEGMENT + LABEL

def load_and_segment(edf_path, seizure_list, selected_channels=None):
    """
    Loads one EDF file, cuts into 5-sec windows,
    extracts features, and assigns labels.
    Returns: (features_array, labels_array, segment_info_list)
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    fs = int(raw.info['sfreq'])

    # Pick channels
    all_ch = raw.ch_names
    if selected_channels:
        use_ch = [c for c in selected_channels if c in all_ch]
    else:
        use_ch = all_ch[:22]  # use first 22 channels like the paper

    data, times = raw[use_ch, :]
    n_samples = data.shape[1]
    n_windows = n_samples // WINDOW_SAMPLES

    all_features = []
    all_labels = []
    segment_info = []   # (start_sample, end_sample, label)

    for w in range(n_windows):
        start = w * WINDOW_SAMPLES
        end = start + WINDOW_SAMPLES
        start_sec = start / fs
        end_sec = end / fs

        # --- LABEL THIS WINDOW ---
        seizure_samples = 0
        for (sz_start, sz_end) in seizure_list:
            overlap_start = max(start_sec, sz_start)
            overlap_end   = min(end_sec,   sz_end)
            if overlap_end > overlap_start:
                seizure_samples += (overlap_end - overlap_start) * fs

        label = 1 if (seizure_samples / WINDOW_SAMPLES) >= SEIZURE_THRESHOLD else 0

        # --- EXTRACT FEATURES FROM EACH CHANNEL ---
        window_feats = []
        for ch_idx in range(len(use_ch)):
            seg = data[ch_idx, start:end]
            feats = extract_features(seg, fs)
            window_feats.extend(feats)

        all_features.append(window_feats)
        all_labels.append(label)
        segment_info.append((start_sec, end_sec, label))

    return (np.array(all_features),
            np.array(all_labels),
            segment_info)


# ============================================================
# PART 4: LOAD FULL SUBJECT DATA
# ============================================================
def load_subject_data(data_dir, seizure_times):
    """
    Loops through all EDF files for one subject.
    Returns combined features + labels.
    """
    all_X, all_y, all_info = [], [], []
    edf_files = sorted([f for f in os.listdir(data_dir)
                        if f.endswith('.edf')])

    for edf_file in edf_files:
        print(f"  Loading {edf_file}...")
        edf_path = os.path.join(data_dir, edf_file)
        sz_list = seizure_times.get(edf_file, [])

        try:
            X, y, info = load_and_segment(edf_path, sz_list)
            all_X.append(X)
            all_y.append(y)
            all_info.extend(info)
        except Exception as e:
            print(f"    Skipping {edf_file}: {e}")
            continue

    if not all_X:
        return None, None, None

    return (np.vstack(all_X),
            np.concatenate(all_y),
            all_info)


# ============================================================
# PART 5: FEATURE SELECTION (Chi-Squared like the paper)
# ============================================================
def select_features(X_train, y_train, X_test, k=32):
    """
    Chi-squared selects the top k features.
    Must scale to [0,1] first since chi2 requires non-negative values.
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Handle NaN/Inf
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=1.0, neginf=0.0)
    X_test_scaled  = np.nan_to_num(X_test_scaled,  nan=0.0, posinf=1.0, neginf=0.0)

    selector = SelectKBest(chi2, k=min(k, X_train_scaled.shape[1]))
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_test_sel  = selector.transform(X_test_scaled)

    return X_train_sel, X_test_sel, selector


# ============================================================
# PART 6: TRAIN RANDOM FOREST
# ============================================================
def train_rf(X_train, y_train):
    """
    Under-sample majority class (non-seizure) to balance training,
    then train Random Forest.
    """
    # Under-sampling: match minority class size
    idx_0 = np.where(y_train == 0)[0]
    idx_1 = np.where(y_train == 1)[0]

    if len(idx_1) == 0:
        print("  WARNING: No seizure segments in training data!")
        return None

    n_minority = len(idx_1)
    idx_0_sampled = np.random.choice(idx_0, size=n_minority, replace=False)
    balanced_idx = np.concatenate([idx_0_sampled, idx_1])
    np.random.shuffle(balanced_idx)

    X_bal = X_train[balanced_idx]
    y_bal = y_train[balanced_idx]

    clf = RandomForestClassifier(
        n_estimators=N_TREES,
        random_state=42,
        n_jobs=-1          # use all CPU cores
    )
    clf.fit(X_bal, y_bal)
    return clf


# ============================================================
# PART 7: POST-PROCESSING FOR EVENT DETECTION
# (Replicates Figure 2 from the paper)
# ============================================================
def post_process_events(y_pred, window_sec=5, gap_sec=10):
    """
    - Fills short non-seizure gaps (<=gap_sec) inside seizure runs
    - Removes short seizure bursts (<=gap_sec) inside non-seizure runs
    Returns: cleaned label array
    """
    gap_windows = int(np.ceil(gap_sec / window_sec))
    y_clean = y_pred.copy()
    n = len(y_clean)

    # Fill short non-seizure gaps between seizures
    i = 0
    while i < n:
        if y_clean[i] == 0:
            j = i
            while j < n and y_clean[j] == 0:
                j += 1
            gap_len = j - i
            # check if surrounded by seizures
            left_ok  = (i > 0 and y_clean[i-1] == 1)
            right_ok = (j < n and y_clean[j] == 1)
            if left_ok and right_ok and gap_len <= gap_windows:
                y_clean[i:j] = 1
            i = j
        else:
            i += 1

    # Remove short isolated seizure bursts
    i = 0
    while i < n:
        if y_clean[i] == 1:
            j = i
            while j < n and y_clean[j] == 1:
                j += 1
            burst_len = j - i
            left_ok  = (i == 0 or y_clean[i-1] == 0)
            right_ok = (j == n or y_clean[j] == 0)
            if left_ok and right_ok and burst_len <= gap_windows:
                y_clean[i:j] = 0
            i = j
        else:
            i += 1

    return y_clean


# ============================================================
# PART 8: EVENT-LEVEL EVALUATION
# (Sen and FDR as defined in the paper)
# ============================================================
def evaluate_events(y_pred_clean, segment_info, seizure_list,
                    total_hours, window_sec=5, overlap_thresh=0.70):
    """
    Matches detected events to true events using 70% overlap rule.
    Returns sensitivity and FDR.
    """
    # Build detected event spans (in seconds)
    detected_events = []
    i = 0
    while i < len(y_pred_clean):
        if y_pred_clean[i] == 1:
            j = i
            while j < len(y_pred_clean) and y_pred_clean[j] == 1:
                j += 1
            det_start = segment_info[i][0]
            det_end   = segment_info[j-1][1]
            detected_events.append((det_start, det_end))
            i = j
        else:
            i += 1

    TP = 0
    FP = len(detected_events)

    for (true_start, true_end) in seizure_list:
        true_dur = true_end - true_start
        best_overlap = 0

        for (det_start, det_end) in detected_events:
            ov_start = max(true_start, det_start)
            ov_end   = min(true_end,   det_end)
            if ov_end > ov_start:
                overlap = (ov_end - ov_start) / true_dur
                best_overlap = max(best_overlap, overlap)

        if best_overlap >= overlap_thresh:
            TP += 1

    FN = len(seizure_list) - TP
    sensitivity = (TP / (TP + FN) * 100) if (TP + FN) > 0 else 0.0

    # FP = detected events that don't match any true event
    matched_det = set()
    for (true_start, true_end) in seizure_list:
        true_dur = true_end - true_start
        for idx, (det_start, det_end) in enumerate(detected_events):
            ov_start = max(true_start, det_start)
            ov_end   = min(true_end,   det_end)
            if ov_end > ov_start:
                if (ov_end - ov_start) / true_dur >= overlap_thresh:
                    matched_det.add(idx)

    fp_count = len(detected_events) - len(matched_det)
    fdr = fp_count / total_hours if total_hours > 0 else 0.0

    return sensitivity, fdr, TP, FN, fp_count


# ============================================================
# PART 9: VISUALISATION
# ============================================================
def plot_results(y_true, y_pred, y_pred_clean, segment_info, subject_name):
    fig, axes = plt.subplots(3, 1, figsize=(15, 8))
    fig.suptitle(f'Seizure Detection — {subject_name}', fontsize=14)

    times = [s[0] / 3600 for s in segment_info]  # convert to hours

    axes[0].fill_between(times, y_true,   alpha=0.7, color='blue',  label='True labels')
    axes[0].set_title('Ground Truth')
    axes[0].set_ylabel('Seizure (1) / Normal (0)')

    axes[1].fill_between(times, y_pred,   alpha=0.7, color='orange', label='Raw predictions')
    axes[1].set_title('Raw Model Predictions')
    axes[1].set_ylabel('Seizure (1) / Normal (0)')

    axes[2].fill_between(times, y_pred_clean, alpha=0.7, color='green', label='Post-processed')
    axes[2].set_title('Post-Processed Event Detection')
    axes[2].set_ylabel('Seizure (1) / Normal (0)')
    axes[2].set_xlabel('Time (hours)')

    for ax in axes:
        ax.set_ylim(-0.1, 1.3)
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f'{subject_name}_results.png', dpi=150)
    plt.show()
    print(f"Plot saved as {subject_name}_results.png")


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Seizure'],
                yticklabels=['Normal', 'Seizure'])
    plt.title('Segment-Level Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()


# ============================================================
# PART 10: MAIN PIPELINE
# Leave-One-Out Cross Validation (as in paper)
# For single-subject demo: simple 80/20 train-test split
# ============================================================
def run_single_subject_demo(data_dir, summary_file):
    print("=" * 60)
    print("EPILEPTIC SEIZURE DETECTION")
    print("=" * 60)

    # 1. Parse seizure times
    print("\n[1] Parsing seizure annotations...")
    seizure_times = parse_summary(summary_file)
    print(f"    Found seizure info for {len(seizure_times)} files")
    total_seizures = sum(len(v) for v in seizure_times.values())
    print(f"    Total seizure events: {total_seizures}")

    # 2. Load all data
    print("\n[2] Loading EEG data and extracting features...")
    print("    (This may take 10–30 minutes for a full subject)")
    X, y, seg_info = load_subject_data(data_dir, seizure_times)

    if X is None:
        print("ERROR: No data loaded. Check your DATA_DIR path.")
        return

    print(f"\n    Total segments: {len(y)}")
    print(f"    Seizure segments: {np.sum(y==1)} "
          f"({np.sum(y==1)/len(y)*100:.2f}%)")
    print(f"    Normal segments:  {np.sum(y==0)} "
          f"({np.sum(y==0)/len(y)*100:.2f}%)")
    print(f"    Class imbalance ratio: 1:{int(np.sum(y==0)/max(np.sum(y==1),1))}")

    # 3. Train/test split (80/20 keeping temporal order)
    print("\n[3] Splitting data (80% train / 20% test)...")
    split_idx = int(len(y) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    info_test = seg_info[split_idx:]

    print(f"    Train: {len(y_train)} segments "
          f"({np.sum(y_train==1)} seizure)")
    print(f"    Test:  {len(y_test)} segments "
          f"({np.sum(y_test==1)} seizure)")

    # 4. Feature selection
    print("\n[4] Selecting features (Chi-Squared)...")
    X_train_sel, X_test_sel, _ = select_features(
        X_train, y_train, X_test, k=N_FEATURES_PER_CHANNEL
    )
    print(f"    Selected {X_train_sel.shape[1]} features")

    # 5. Train RF
    print("\n[5] Training Random Forest...")
    clf = train_rf(X_train_sel, y_train)
    if clf is None:
        print("Training failed — not enough seizure data.")
        return
    print(f"    Trained with {N_TREES} trees")

    # 6. Predict
    print("\n[6] Running predictions on test set...")
    y_pred = clf.predict(X_test_sel)

    # 7. Segment-level results
    print("\n[7] SEGMENT-LEVEL RESULTS:")
    print("-" * 40)
    print(classification_report(y_test, y_pred,
                                target_names=['Normal', 'Seizure'],
                                zero_division=0))
    seg_sensitivity = recall_score(y_test, y_pred,
                                   pos_label=1, zero_division=0)
    print(f"    Sensitivity (recall): {seg_sensitivity*100:.2f}%")

    # 8. Post-processing
    print("\n[8] Post-processing for event detection...")
    y_pred_clean = post_process_events(y_pred)

    # 9. Event-level evaluation
    print("\n[9] EVENT-LEVEL RESULTS:")
    print("-" * 40)
    # Collect all seizure events from test portion
    test_start_sec = info_test[0][0] if info_test else 0
    test_end_sec   = info_test[-1][1] if info_test else 0
    total_test_hours = (test_end_sec - test_start_sec) / 3600

    # Gather seizures that fall in test window
    test_seizures = []
    for sz_list in seizure_times.values():
        for (s, e) in sz_list:
            if s >= test_start_sec or e >= test_start_sec:
                test_seizures.append((s, e))

    sensitivity, fdr, tp, fn, fp = evaluate_events(
        y_pred_clean, info_test, test_seizures,
        total_test_hours
    )
    print(f"    True Positives (detected events): {tp}")
    print(f"    False Negatives (missed events):  {fn}")
    print(f"    False Detections:                 {fp}")
    print(f"    Sensitivity:  {sensitivity:.2f}%")
    print(f"    FDR:          {fdr:.2f} false detections/hour")
    print(f"    Test duration: {total_test_hours:.2f} hours")

    # 10. Visualise
    print("\n[10] Generating plots...")
    plot_confusion_matrix(y_test, y_pred)
    plot_results(y_test, y_pred, y_pred_clean,
                 info_test, "chb01")

    print("\n✅ Done! Check the generated PNG files.")


# ============================================================
# RUN IT
# ============================================================
if __name__ == "__main__":
    run_single_subject_demo(DATA_DIR, SUMMARY_FILE)