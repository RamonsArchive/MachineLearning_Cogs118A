import os
import sys
import json

# Add src directory to Python path so imports work from any directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.dont_write_bytecode = True  # Prevent __pycache__ creation


def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    main()