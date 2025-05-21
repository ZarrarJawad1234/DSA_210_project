import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import wfdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt, medfilt
import random
from datetime import datetime
from collections import Counter

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Set up directories
DATA_DIR = Path("C:/Users/zarra_8gw2s8r/ECGProjectPTB/data2")   
MODEL_DIR = DATA_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
PTB_PATH = Path("C:/Users/zarra_8gw2s8r/ECGProjectPTB/data/ptb-diagnostic-ecg-database-1.0.0")

# Official PTB diagnosis mapping
DIAGNOSIS_MAP = {
    'healthy control': 'Normal',
    'myocardial infarction': 'Defective',
    'cardiomyopathy': 'Defective',
    'bundle branch block': 'Defective',
    'dysrhythmia': 'Defective',
    'myocardial hypertrophy': 'Defective',
    'valvular heart disease': 'Defective',
    'myocarditis': 'Defective',
    'miscellaneous': 'Defective',
    'heart failure': 'Defective',
    'hypertension': 'Defective'
}

# --- Data Processing ---

def preprocess_signal(signal, fs=1000):
    """Enhanced ECG signal preprocessing pipeline"""
    baseline = medfilt(signal, kernel_size=501)
    signal = signal - baseline
    low = 0.5
    high = 40
    nyq = 0.5 * fs
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    signal = filtfilt(b, a, signal)
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    return signal

def get_ptb_diagnosis(record):
    """Get official PTB diagnosis from record comments"""
    comments = ' '.join(record.comments).lower()
    for key in DIAGNOSIS_MAP:
        if key in comments:
            return DIAGNOSIS_MAP[key]
    return 'Defective'

def load_ptb_data(max_records=None):
    signals = []
    hea_files = list(PTB_PATH.rglob("*.hea"))
    if not hea_files:
        print(f"Error: No .hea files found in {PTB_PATH}.")
        return pd.DataFrame()
    max_records = len(hea_files) if max_records is None else min(max_records, len(hea_files))
    for i, record_path in enumerate(hea_files[:max_records]):
        record_name = record_path.stem
        try:
            record = wfdb.rdrecord(str(record_path.parent / record_name))
            signal = record.p_signal[:, 0][:5000]
            if len(signal) == 5000:
                diagnosis = get_ptb_diagnosis(record)
                signal = preprocess_signal(signal)
                signals.append({
                    "patient_id": record_name,
                    "signal": signal.tolist(),
                    "diagnosis": diagnosis
                })
        except Exception as e:
            print(f"Error processing {record_name}: {str(e)}")
    signals_df = pd.DataFrame(signals)
    return signals_df

# --- Model Architecture ---

class ECGClassifier(nn.Module):
    def __init__(self):
        super(ECGClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            nn.Conv1d(16, 32, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# --- Dataset ---

class ECGDataset(Dataset):
    def __init__(self, signals, labels, augment=False):
        self.signals = signals
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        if self.augment and random.random() > 0.5:
            signal = signal + np.random.normal(0, 0.01, signal.shape)
            signal = signal * random.uniform(0.95, 1.05)
        return torch.tensor(signal, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.float32)

# --- Training and Evaluation ---

def get_class_weights(labels):
    class_counts = Counter(labels)
    total_samples = sum(class_counts.values())
    return torch.tensor([total_samples / class_counts[0], total_samples / class_counts[1]], dtype=torch.float32)

def train_model(model, train_loader, test_loader, device, epochs=15):
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.cpu().numpy())
    class_weights = get_class_weights(all_labels).to(device)
    criterion = nn.BCELoss(weight=class_weights[1] * 1.5)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    best_val_loss = float('inf')
    patience = 3
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_recall_normal': [], 'val_recall_defective': []}
    thresholds = np.linspace(0.4, 0.8, 50)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        model.eval()
        val_loss, correct, total = 0, 0, 0
        all_val_labels = []
        all_val_probs = []
        with torch.no_grad():
            for signals, labels in test_loader:
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                val_loss += criterion(outputs.squeeze(), labels).item()
                correct += (outputs.squeeze().round() == labels).sum().item()
                total += labels.size(0)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_probs.extend(outputs.cpu().numpy())
        val_loss /= len(test_loader)
        val_acc = correct / total
        # Optimize threshold for balanced recall and F1-score
        recall_normal_scores = [recall_score(all_val_labels, (np.array(all_val_probs) > t).astype(int), zero_division=0) for t in thresholds]
        recall_defective_scores = [recall_score(1 - np.array(all_val_labels), 1 - (np.array(all_val_probs) > t).astype(int), zero_division=0) for t in thresholds]
        f1_scores = [f1_score(all_val_labels, (np.array(all_val_probs) > t).astype(int), zero_division=0) for t in thresholds]
        combined_scores = [0.5 * normal_recall + 0.3 * f1 + 0.2 * defective_recall for normal_recall, defective_recall, f1 in zip(recall_normal_scores, recall_defective_scores, f1_scores)]
        optimal_threshold = thresholds[np.argmax(combined_scores)]
        val_preds = (np.array(all_val_probs) > optimal_threshold).astype(int)
        val_f1 = f1_score(all_val_labels, val_preds, zero_division=0)
        val_recall_normal = recall_score(all_val_labels, val_preds, zero_division=0)
        val_recall_defective = recall_score(1 - np.array(all_val_labels), 1 - val_preds, zero_division=0)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_recall_normal'].append(val_recall_normal)
        history['val_recall_defective'].append(val_recall_defective)
        print(f"Epoch {epoch + 1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}, "
              f"Val F1: {val_f1:.4f}, "
              f"Val Normal Recall: {val_recall_normal:.4f}, "
              f"Val Defective Recall: {val_recall_defective:.4f}, "
              f"Threshold: {optimal_threshold:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    return history, optimal_threshold

def evaluate_model(model, test_loader, device, threshold=0.5):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for signals, labels in test_loader:
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
            probs = outputs.cpu().numpy()
            preds = (probs > threshold).astype(int)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Defective", "Normal"], zero_division=0))
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Defective", "Normal"],
                yticklabels=["Defective", "Normal"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(DATA_DIR / "confusion_matrix.png")
    plt.close()
    print(f"Saved plot: {DATA_DIR / 'confusion_matrix.png'}")
    return all_labels, all_preds, all_probs

# --- Plot and Test ECGs ---

def plot_normal_defective_ecg(normal_signal, defective_signal):
    plt.figure(figsize=(10, 6))
    time = np.arange(len(normal_signal)) / 1000
    plt.plot(time, normal_signal, label="Normal", color="blue")
    plt.plot(time, defective_signal, label="Defective", color="red")
    plt.title("Normal vs. Defective ECG")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.savefig(DATA_DIR / "normal_vs_defective_ecg.png")
    plt.close()
    print(f"Saved ECG plot: {DATA_DIR / 'normal_vs_defective_ecg.png'}")

def test_random_ecg(model, signal, diagnosis, label_encoder, device, filename, threshold=0.5):
    signal_tensor = torch.tensor(signal.reshape(1, 1, -1), dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(signal_tensor)
        prob = output.cpu().numpy()[0].item()
        pred = 1 if prob > threshold else 0
    prediction = label_encoder.inverse_transform([pred])[0]
    true_label = label_encoder.inverse_transform([int(diagnosis)])[0]
    print(f"\n{filename.replace('.png', '')}:")
    print(f"Probability (Normal): {prob:.6f}")
    print(f"Prediction: {prediction}")
    print(f"True Label: {true_label}")
    plt.figure(figsize=(10, 6))
    time = np.arange(len(signal)) / 1000
    plt.plot(time, signal, label="ECG", color="blue" if true_label == "Normal" else "red")
    plt.title(f"{filename.replace('.png', '')} (Pred: {prediction}, True: {true_label})")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.savefig(DATA_DIR / filename)
    plt.close()
    print(f"Saved ECG plot: {DATA_DIR / filename}")
    return prediction == true_label

# --- Main Execution ---

if __name__ == "__main__":
    print("\n--- Loading and Preprocessing Data ---")
    signals_df = load_ptb_data()
    if signals_df.empty:
        print("Error: No data loaded. Exiting.")
        exit()
    master_df = signals_df
    master_df.to_csv(DATA_DIR / "master_ecg_data.csv", index=False)
    print("\nClass Distribution:")
    print(master_df["diagnosis"].value_counts())
    label_encoder = LabelEncoder()
    master_df["label"] = label_encoder.fit_transform(master_df["diagnosis"])
    print(f"\nLabel mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    train_df, test_df = train_test_split(
        master_df,
        test_size=0.2,
        stratify=master_df["label"],
        random_state=42
    )
    X_train = np.array([np.array(s) for s in train_df["signal"]])
    X_test = np.array([np.array(s) for s in test_df["signal"]])
    y_train = train_df["label"].values
    y_test = test_df["label"].values
    train_dataset = ECGDataset(X_train, y_train, augment=True)
    test_dataset = ECGDataset(X_test, y_test)
    class_weights = 1. / torch.tensor([(y_train == 0).sum(), (y_train == 1).sum()], dtype=torch.float32)
    sample_weights = class_weights[y_train] * 0.5
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    device = torch.device("cpu")
    print(f"\nUsing device: {device}")
    model = ECGClassifier().to(device)
    print("\n--- Training Model ---")
    history, optimal_threshold = train_model(model, train_loader, test_loader, device, epochs=22)
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pth"))
    print("\n--- Evaluating Model ---")
    all_labels, all_preds, all_probs = evaluate_model(model, test_loader, device, threshold=optimal_threshold)
    normal_idx = np.where(y_test == label_encoder.transform(["Normal"])[0])[0]
    defective_idx = np.where(y_test == label_encoder.transform(["Defective"])[0])[0]
    if len(normal_idx) > 0 and len(defective_idx) > 0:
        normal_signal = X_test[normal_idx[0]]
        defective_signal = X_test[defective_idx[0]]
        plot_normal_defective_ecg(normal_signal, defective_signal)
        # Test three random Normal ECGs
        random_normal_indices = random.sample(list(normal_idx), min(3, len(normal_idx)))
        for i, idx in enumerate(random_normal_indices):
            test_random_ecg(
                model, X_test[idx], y_test[idx],
                label_encoder, device, f"random_normal_ecg_{i+1}.png", threshold=optimal_threshold
            )
        # Test one random Defective ECG
        random_defective_idx = random.choice(defective_idx)
        test_random_ecg(
            model, X_test[random_defective_idx], y_test[random_defective_idx],
            label_encoder, device, "random_defective_ecg.png", threshold=optimal_threshold
        )
    else:
        print("Not enough normal or defective samples to plot or test ECGs.")
    print("\nProcessing complete. Check C:/Users/zarra_8gw2s8r/ECGProjectPTB/data2 for outputs.")