import argparse
import os

def main(args):
    # Simulating model registration
    model_info = {
        'model_name': args.model_name,
        'model_path': args.model_path,
        'evaluation_output': args.evaluation_output
    }
    
    # Save the model information to a file
    os.makedirs(os.path.dirname(args.model_info_output_path), exist_ok=True)
    with open(args.model_info_output_path, 'w') as f:
        for key, value in model_info.items():
            f.write(f'{key}: {value}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Name of the model")
    parser.add_argument("--model_path", type=str, help="Path to the trained model")
    parser.add_argument("--evaluation_output", type=str, help="Path to the evaluation results")
    parser.add_argument("--model_info_output_path", type=str, help="Path to save the model info")
    
    args = parser.parse_args()
    main(args)
