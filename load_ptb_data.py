import pandas as pd
import numpy as np
import wfdb
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for saving plots in PyCharm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import ttest_ind, chi2_contingency

# Set up directories
DATA_DIR = Path("C:/Users/zarra_8gw2s8r/ECGProjectPTB/data2")
DATA_DIR.mkdir(exist_ok=True)
PTB_PATH = Path("C:/Users/zarra_8gw2s8r/ECGProjectPTB/data/ptb-diagnostic-ecg-database-1.0.0")


# Extract ECG signals and demographics from PTB Diagnostic (process all records)
def load_ptb_data(max_records=None):
    signals = []
    demographics = []

    if not PTB_PATH.exists():
        print(f"Error: Directory {PTB_PATH} does not exist.")
        print(f"Current working directory: {Path.cwd()}")
        print(f"Expected data path: {PTB_PATH.resolve()}")
        return pd.DataFrame(), pd.DataFrame()

    hea_files = list(PTB_PATH.rglob("*.hea"))
    if not hea_files:
        print(f"Error: No .hea files found in {PTB_PATH}.")
        return pd.DataFrame(), pd.DataFrame()

    print(f"Found {len(hea_files)} .hea files in {PTB_PATH}.")
    max_records = len(hea_files) if max_records is None else min(max_records, len(hea_files))

    for i, record_path in enumerate(hea_files):
        if i >= max_records:
            break
        record_name = record_path.stem
        try:
            # Extract signal
            record = wfdb.rdrecord(str(record_path.parent / record_name))
            signal = record.p_signal[:, 0][:5000]  # Lead I, 5000 samples (5s at 1000 Hz)
            if len(signal) == 5000:
                # Parse diagnosis
                diagnosis = "Defective"
                for comment in record.comments:
                    if "reason for admission: myocardial infarction" in comment.lower():
                        diagnosis = "Defective"
                        break
                    if "normal" in comment.lower():
                        diagnosis = "Normal"
                        break
                signals.append({
                    "patient_id": record_name,
                    "signal": signal.tolist(),
                    "diagnosis": diagnosis,
                    "mean_signal": np.mean(signal)  # Compute mean signal amplitude
                })
                # Extract demographics
                age = None
                gender = None
                for comment in record.comments:
                    comment_lower = comment.lower()
                    if "age" in comment_lower:
                        try:
                            age_part = comment_lower.split("age")[-1].strip()
                            age = int(age_part.split(":")[-1].strip())
                        except (IndexError, ValueError):
                            continue
                    if "sex" in comment_lower:
                        try:
                            gender_part = comment_lower.split("sex")[-1].strip()
                            gender = gender_part.split(":")[-1].strip().upper()
                            gender = "M" if gender == "MALE" else "F" if gender == "FEMALE" else None
                        except (IndexError, ValueError):
                            continue
                demographics.append({
                    "patient_id": record_name,
                    "age": age,
                    "gender": gender
                })
            else:
                print(f"Signal length for {record_name} is {len(signal)}, expected 5000. Skipping.")
        except Exception as e:
            print(f"Error processing {record_name}: {e}")

    # Create DataFrames
    signals_df = pd.DataFrame(signals)
    demo_df = pd.DataFrame(demographics)

    # Handle missing demographics
    if not demographics:
        print("No demographics data extracted. Check .hea files for comments.")
        demo_df["age"] = pd.Series(dtype=float)
        demo_df["gender"] = pd.Series(dtype=str)
    else:
        if "age" in demo_df.columns:
            demo_df["age"] = demo_df["age"].fillna(demo_df["age"].mean())
        else:
            print("No 'age' data found in comments. Adding empty age column.")
            demo_df["age"] = pd.Series(dtype=float)

    return signals_df, demo_df


# Merge and save data
def merge_data(signals_df, demo_df):
    if signals_df.empty or demo_df.empty:
        print("One or both DataFrames are empty. Cannot merge.")
        return pd.DataFrame()
    master_df = signals_df.merge(demo_df, on="patient_id", how="inner")
    master_df.to_csv(DATA_DIR / "master_ecg_data.csv", index=False)
    return master_df


# Analyze data, perform hypothesis testing, and create a visualization for hypothesis testing results
def analyze_data(master_df):
    if master_df.empty:
        print("Master DataFrame is empty. Cannot analyze data.")
        return

    # Summary statistics
    print("Summary Statistics:")
    print(master_df.describe(include="all"))

    # Print full DataFrames
    print("\nSignals DataFrame (full dataset):")
    print(signals_df)
    print("\nDemographics DataFrame (full dataset):")
    print(demo_df)
    print("\nMerged DataFrame (full dataset):")
    print(master_df)

    # # Age distribution by diagnosis (Boxplot and Histogram) - Commented out
    # print("\nAge Distribution by Diagnosis:")
    # plt.figure(figsize=(8, 6))
    # sns.boxplot(x="diagnosis", y="age", hue="diagnosis", data=master_df, palette=["teal", "coral"])
    # plt.title("Age Distribution by Diagnosis")
    # plt.xlabel("Diagnosis")
    # plt.ylabel("Age")
    # try:
    #     plt.savefig(DATA_DIR / "age_by_diagnosis_boxplot.png")
    #     print(f"Saved plot: {DATA_DIR / 'age_by_diagnosis_boxplot.png'}")
    # except Exception as e:
    #     print(f"Error saving age by diagnosis boxplot: {e}")
    # plt.close()

    # plt.figure(figsize=(8, 6))
    # sns.histplot(data=master_df, x="age", hue="diagnosis", element="step", stat="count", common_norm=False, palette=["teal", "coral"])
    # plt.title("Age Distribution by Diagnosis")
    # plt.xlabel("Age")
    # plt.ylabel("Count")
    # try:
    #     plt.savefig(DATA_DIR / "age_by_diagnosis_histogram.png")
    #     print(f"Saved plot: {DATA_DIR / 'age_by_diagnosis_histogram.png'}")
    # except Exception as e:
    #     print(f"Error saving age by diagnosis histogram: {e}")
    # plt.close()

    # # Gender vs. Diagnosis Crosstab (Heatmap) - Commented out
    # print("\nGender vs. Diagnosis Crosstab:")
    # gender_diagnosis = pd.crosstab(master_df["gender"], master_df["diagnosis"])
    # print(gender_diagnosis)
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(gender_diagnosis, annot=True, fmt="d", cmap="YlGnBu")
    # plt.title("Gender vs. Diagnosis Heatmap")
    # plt.xlabel("Diagnosis")
    # plt.ylabel("Gender")
    # try:
    #     plt.savefig(DATA_DIR / "gender_diagnosis_heatmap.png")
    #     print(f"Saved plot: {DATA_DIR / 'gender_diagnosis_heatmap.png'}")
    # except Exception as e:
    #     print(f"Error saving gender vs. diagnosis heatmap: {e}")
    # plt.close()

    # # Gender distribution (Pie Chart) - Commented out
    # print("\nGender Distribution:")
    # gender_counts = master_df["gender"].value_counts()
    # print(gender_counts)
    # plt.figure(figsize=(8, 6))
    # plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=["blue", "pink"])
    # plt.title("Gender Distribution")
    # try:
    #     plt.savefig(DATA_DIR / "gender_distribution.png")
    #     print(f"Saved plot: {DATA_DIR / 'gender_distribution.png'}")
    # except Exception as e:
    #     print(f"Error saving gender distribution plot: {e}")
    # plt.close()

    # # Scatterplot: Age vs. Mean Signal Amplitude by Diagnosis - Commented out
    # plt.figure(figsize=(8, 6))
    # sns.scatterplot(data=master_df, x="age", y="mean_signal", hue="diagnosis", style="gender", palette=["blue", "orange"], s=100)
    # plt.title("Scatterplot: Age vs. Mean Signal Amplitude by Diagnosis")
    # plt.xlabel("Age")
    # plt.ylabel("Mean Signal Amplitude")
    # try:
    #     plt.savefig(DATA_DIR / "scatter_age_mean_signal.png")
    #     print(f"Saved plot: {DATA_DIR / 'scatter_age_mean_signal.png'}")
    # except Exception as e:
    #     print(f"Error saving scatterplot (age vs. mean signal): {e}")
    # plt.close()

    # # Pairplot: Relationships between Age, Mean Signal, and Diagnosis - Commented out
    # plt.figure(figsize=(10, 8))
    # sns.pairplot(master_df[["age", "mean_signal", "diagnosis"]], hue="diagnosis", palette=["blue", "orange"], height=2.5)
    # try:
    #     plt.savefig(DATA_DIR / "pairplot_age_signal_diagnosis.png")
    #     print(f"Saved plot: {DATA_DIR / 'pairplot_age_signal_diagnosis.png'}")
    # except Exception as e:
    #     print(f"Error saving pairplot: {e}")
    # plt.close()

    # Hypothesis Testing
    print("\n--- Hypothesis Testing ---")

    # Hypothesis 1: Age difference between Normal and Defective ECGs (t-test)
    print("\nHypothesis 1: Age Difference Between Normal and Defective ECGs")
    age_normal = master_df[master_df["diagnosis"] == "Normal"]["age"].dropna()
    age_defective = master_df[master_df["diagnosis"] == "Defective"]["age"].dropna()
    t_stat_age, p_value_age = ttest_ind(age_normal, age_defective, equal_var=False)  # Welch's t-test
    print(f"T-statistic: {t_stat_age:.4f}, P-value: {p_value_age:.4f}")
    if p_value_age < 0.05:
        print("Result: Reject the null hypothesis - significant difference in age between normal and defective ECGs.")
    else:
        print(
            "Result: Fail to reject the null hypothesis - no significant difference in age between normal and defective ECGs.")

    # Hypothesis 2: Gender and Diagnosis Association (Chi-square test)
    print("\nHypothesis 2: Gender and Diagnosis Association")
    gender_diagnosis = pd.crosstab(master_df["gender"], master_df["diagnosis"])
    chi2_stat, p_value_gender, dof, expected = chi2_contingency(gender_diagnosis)
    print(f"Chi-square statistic: {chi2_stat:.4f}, P-value: {p_value_gender:.4f}, Degrees of Freedom: {dof}")
    print("Expected frequencies:")
    print(expected)
    if p_value_gender < 0.05:
        print("Result: Reject the null hypothesis - significant association between gender and diagnosis.")
    else:
        print("Result: Fail to reject the null hypothesis - no significant association between gender and diagnosis.")

    # Hypothesis 3: Mean Signal Amplitude Difference by Diagnosis (t-test)
    print("\nHypothesis 3: Mean Signal Amplitude Difference by Diagnosis")
    signal_normal = master_df[master_df["diagnosis"] == "Normal"]["mean_signal"].dropna()
    signal_defective = master_df[master_df["diagnosis"] == "Defective"]["mean_signal"].dropna()
    t_stat_signal, p_value_signal = ttest_ind(signal_normal, signal_defective, equal_var=False)  # Welch's t-test
    print(f"T-statistic: {t_stat_signal:.4f}, P-value: {p_value_signal:.4f}")
    if p_value_signal < 0.05:
        print(
            "Result: Reject the null hypothesis - significant difference in mean signal amplitude between normal and defective ECGs.")
    else:
        print(
            "Result: Fail to reject the null hypothesis - no significant difference in mean signal amplitude between normal and defective ECGs.")

    # Hypothesis Testing Chart: Bar Chart of P-values
    p_values = {
        "Age Difference (Normal vs. Defective)": p_value_age,
        "Gender vs. Diagnosis Association": p_value_gender,
        "Mean Signal Amplitude (Normal vs. Defective)": p_value_signal
    }
    plt.figure(figsize=(10, 6))
    plt.bar(p_values.keys(), p_values.values(), color="skyblue")
    plt.axhline(y=0.05, color="red", linestyle="--", label="Significance Threshold (0.05)")
    plt.title("Hypothesis Testing P-values")
    plt.ylabel("P-value")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.yscale("log")  # Use log scale for better visualization of small p-values
    plt.tight_layout()
    try:
        plt.savefig(DATA_DIR / "hypothesis_testing_pvalues.png")
        print(f"Saved plot: {DATA_DIR / 'hypothesis_testing_pvalues.png'}")
    except Exception as e:
        print(f"Error saving hypothesis testing p-values plot: {e}")
    plt.close()


# Main execution
if __name__ == "__main__":
    # Load data (process all records)
    signals_df, demo_df = load_ptb_data(max_records=None)

    # Save to CSVs
    if not signals_df.empty:
        signals_df.to_csv(DATA_DIR / "ecg_data.csv", index=False)
    if not demo_df.empty:
        demo_df.to_csv(DATA_DIR / "demographics.csv", index=False)

    # Merge data
    master_df = merge_data(signals_df, demo_df)

    # Analyze data and perform hypothesis testing
    analyze_data(master_df)

    print("Analysis complete. Check C:/Users/zarra_8gw2s8r/ECGProjectPTB/data2 for the hypothesis testing chart.")