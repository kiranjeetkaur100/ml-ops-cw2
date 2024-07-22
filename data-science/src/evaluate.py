import pandas as pd
from sklearn.metrics import accuracy_score
import joblib
import argparse
import os

def main(args):
    # Load the test data
    test_data = pd.read_csv(args.test_data)
    
    # Load the trained model
    model = joblib.load(args.model_input)
    
    # Make predictions on the test data
    predictions = model.predict(test_data['Message'])
    
    # Calculate accuracy
    accuracy = accuracy_score(test_data['Spam'], predictions)
    
    # Save the evaluation result
    os.makedirs(os.path.dirname(args.evaluation_output), exist_ok=True)
    with open(args.evaluation_output, 'w') as f:
        f.write(f'Accuracy: {accuracy}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_input", type=str, help="Path to the trained model")
    parser.add_argument("--test_data", type=str, help="Path to the test data")
    parser.add_argument("--evaluation_output", type=str, help="Path to save the evaluation results")
    
    args = parser.parse_args()
    main(args)
