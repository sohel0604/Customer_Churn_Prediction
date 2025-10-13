from setuptools import setup, find_packages

setup(
    name="Telco_Churn_Prediction",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib",
        "xgboost",
    ],
)
