import pandas as pd
import numpy as np


""" This function cleans the bank data by removing missing values and unknown values. 
    No missing (null) values so I return the data as is, keeping the unkown values as is.
    about 10% of the data (rows) has unknown values.
"""
def clean_bank(data):
    # print("\n === DATA SHAPE ===")
    # print(data.shape)
    # print("Number of missing values:")
    # print(data.isnull().sum().sum())
    # print("Number of missing values per column:")
    # print(data.isnull().sum())
    # print("Number of missing values per row:")
    # print(data.isnull().sum(axis=1))

    # print("\n=== Unkown values ===")
    # print((data == "unknown").sum())

    # print("\n=== ROWS containing unknown values ===") 
    # print(data[data.eq("unknown").any(axis=1)])

    # print("\n=== LOCATIONS OF UNKNOWN ===")
    # rows, cols = np.where(data.eq("unknown"))
    # for r, c in zip(rows, cols):
    #     print(f"Row {r}, Column '{data.columns[c]}'")



    exclude_cols = ["duration", "campaign"]  # Clear data leakage
    df = data.drop(columns=exclude_cols)
    return df
