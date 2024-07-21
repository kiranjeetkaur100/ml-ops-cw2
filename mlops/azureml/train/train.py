import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import argparse

def main(args):
    # Load dataset
    data = pd.read_csv(args.data_path)
    
    # Preprocess dataset
    data['Spam'] = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(data.Message, data.Spam, test_size=0.25, random_state=42)
    
    # Create a pipeline
    clf = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('nb', MultinomialNB())
    ])
    
    # Train the model
    clf.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = clf.score(X_test, y_test)
    print(f'Accuracy: {accuracy}')
    
    # Save the model
    joblib.dump(clf, 'spam_model.pkl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing the dataset')
    args = parser.parse_args()
    main(args)
