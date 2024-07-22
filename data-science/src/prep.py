import pandas as pd
import numpy as np
import argparse
import os

def main(args):
    data = pd.read_csv(args.raw_data)
    data['Spam'] = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)
    
    # Split the data into train, validation, and test sets
    train, val, test = np.split(data.sample(frac=1, random_state=42), [int(.7*len(data)), int(.85*len(data))])

    # Save the datasets to the specified output paths
    os.makedirs(os.path.dirname(args.train_data), exist_ok=True)
    os.makedirs(os.path.dirname(args.val_data), exist_ok=True)
    os.makedirs(os.path.dirname(args.test_data), exist_ok=True)

    train.to_csv(args.train_data, index=False)
    val.to_csv(args.val_data, index=False)
    test.to_csv(args.test_data, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", type=str, help="Path to the raw data file")
    parser.add_argument("--train_data", type=str, help="Path to save the training data")
    parser.add_argument("--val_data", type=str, help="Path to save the validation data")
    parser.add_argument("--test_data", type=str, help="Path to save the test data")
    parser.add_argument("--enable_monitoring", type=str, help="Enable monitoring flag")
    parser.add_argument("--table_name", type=str, help="Table name for monitoring")

    args = parser.parse_args()
    main(args)
