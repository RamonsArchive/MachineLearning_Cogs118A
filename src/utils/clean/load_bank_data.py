import pandas as pd
import os
def load_bank_data(curr_dir):
    data = pd.read_csv(os.path.join(curr_dir, "../../datasets/BankSets/bank-additional/bank-additional-full.csv"))
    return data