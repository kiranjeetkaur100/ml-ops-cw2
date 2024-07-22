import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import argparse
import os

def main(args):
    # Load training data
    train_data = pd.read_csv(args.train_data)
    
    # Create a text processing and machine learning pipeline
    clf = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('nb', MultinomialNB())
    ])
    
    # Train the model
    clf.fit(train_data['Message'], train_data['Spam'])
    
    # Save the trained model
    os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
    joblib.dump(clf, args.model_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="Path to the training data")
    parser.add_argument("--model_output", type=str, help="Path to save the trained model")
    
    args = parser.parse_args()
    main(args)
