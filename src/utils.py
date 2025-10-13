import os
import joblib
import json
import pandas as pd

def save_object(obj, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(obj, filepath)

def load_object(filepath):
    return joblib.load(filepath)

def save_df(df, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def save_json(obj, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(obj, f, indent=2)
