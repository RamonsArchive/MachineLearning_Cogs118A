import os
import sys
from utils.clean.load_bank_data import load_bank_data
def main():
    print("Hello, World!")

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data = load_bank_data(curr_dir)
    print(data.head())
    
if __name__ == "__main__":
    main()

