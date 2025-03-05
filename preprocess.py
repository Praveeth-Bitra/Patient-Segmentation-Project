import pandas as pd
import os

data_dir = r"C:\PROJECT-1\Research1"

admissions = pd.read_csv(os.path.join(data_dir, "ADMISSIONS.csv"), low_memory=False)
patients = pd.read_csv(os.path.join(data_dir, "PATIENTS.csv"), low_memory=False)
diagnoses = pd.read_csv(os.path.join(data_dir, "DIAGNOSES_ICD.csv"), low_memory=False)

admissions["admittime"] = pd.to_datetime(admissions["admittime"])
admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])
admissions["deathtime"] = pd.to_datetime(admissions["deathtime"])
patients["dob"] = pd.to_datetime(patients["dob"])
patients["dod"] = pd.to_datetime(patients["dod"])

admissions_cleaned = admissions.drop(columns=["row_id", "edregtime", "edouttime"])
patients_cleaned = patients.drop(columns=["row_id", "dod_hosp", "dod_ssn"])
diagnoses_cleaned = diagnoses.drop(columns=["row_id", "seq_num"])

merged_df = admissions_cleaned.merge(patients_cleaned, on="subject_id", how="inner")
merged_df = merged_df.merge(diagnoses_cleaned, on=["subject_id", "hadm_id"], how="left")

merged_df["icd9_code"].fillna("Unknown", inplace=True)
merged_df.fillna("Unknown", inplace=True)

cleaned_data_path = os.path.join(data_dir, "CLEANED_MIMIC.csv")
merged_df.to_csv(cleaned_data_path, index=False)

print(f"Data cleaning complete. Cleaned data saved to {cleaned_data_path}")
print(merged_df.head())
from sklearn.preprocessing import LabelEncoder

cleaned_data_path = os.path.join(data_dir, "CLEANED_MIMIC.csv")
df = pd.read_csv(cleaned_data_path)


features = ["age", "gender", "admission_type", "insurance", "ethnicity", "icd9_code"]
df["age"] = (pd.to_datetime("2100-01-01") - pd.to_datetime(df["dob"])).dt.days // 365  
df = df[features]

label_encoders = {}
for col in ["gender", "admission_type", "insurance", "ethnicity", "icd9_code"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le 
processed_data_path = os.path.join(data_dir, "PROCESSED_MIMIC.csv")
df.to_csv(processed_data_path, index=False)

print(f"Feature selection complete. Processed data saved to {processed_data_path}")
print(df.head())
