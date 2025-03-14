# DSA_210_project
Zarrar Jawad

35027

Motivation
ECG analysis is a key tool for catching cardiac issues early, and I want to use data science to make it even better—potentially helping doctors spot problems faster and save lives. This project will explore how ECG signal patterns, paired with patient details like age, can help tell apart normal heartbeats from defective ones. I’ll be working with public ECG datasets and adding demographic info to dig into patterns, like whether age plays a role in heart issues. I actually tried a similar project before but didn’t get far, so I’m really motivated to make it work this time with the skills I’m learning in DSA 210.

Data Sources
I’ll be pulling my data from PhysioNet, a go-to place for medical research datasets. My main source will be the PTB Diagnostic ECG Database and the ECG 5000, which has over 14,000 ECG recordings from 290 people, labeled as normal or showing issues like heart attacks or irregular rhythms. It also comes with patient info like age and gender, which I’ll use to enrich my analysis. For a backup and to cross-check my findings, I’ll also grab some data from the MIT-BIH Arrhythmia Database, which has 48 half-hour ECG recordings with detailed beat-by-beat labels. Both datasets are available on PhysioNet, and I’ll focus on the ECG signals and patient demographics to see how they tie into heart conditions.

Data Collection Plan
Here’s how I’ll get my data ready: I’ll start by downloading the PTB Diagnostic ECG Database from PhysioNet (https://physionet.org/content/ptbdb/1.0.0/) on March 11, 2025. The files come in a special WFDB format, with signal data and headers that include things like sampling rate and labels. I’ll use the wfdb-python package in Python to convert these into a CSV file called ecg_data.csv by March 15. I’ll focus on the first 10 seconds of each recording—about 5,000 samples at 500 Hz—and stick to lead I for consistency, labeling each as “Normal” or “Defective” based on the diagnosis.

For the demographic info, I’ll pull age, gender, and any clinical notes from the PTB metadata files, also on PhysioNet. I’ll write a script (extract_metadata.py) to turn this into a demographics.csv file by March 12, making sure ages are numbers (like 65 instead of “65 years”) and gender is either M or F. If some ages are missing, I’ll use the average age from the dataset, which is around 55.

To double-check my work I’ll use wfdb-python again to convert a small subset into mitbih_data.csv, picking out 5,000 beats to compare with my main dataset. Then, I’ll merge everything into a single master_ecg_data.csv file using pandas, matching ECG records with their demographic info by patient ID and dropping any incomplete rows. I’ll make sure the signals are clean by filtering out weird outliers, like values way off the charts, and I’ll check the labels against PhysioNet’s documentation to avoid mistakes. All my files will go into a /data folder in my GitHub repo, and I’ll commit my progress daily from March 11 to 15.

Tools:

I’ll stick with Python 3.11 for the whole project, using pandas to wrangle the data, wfdb-python to handle PhysioNet files, scikit-learn for the machine learning part, and matplotlib for creating those cool visualizations. My GitHub repo will keep everything organized and ready for submission.
