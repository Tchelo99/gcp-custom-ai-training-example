import argparse
import joblib
import os
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train_model(args):
    """Trains a simple logistic regression model."""

    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LogisticRegression(solver='lbfgs', C=args.C)
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy}")

    # Save the model
    if args.model_dir:
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
        print(f"Model saved to {args.model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("AIP_MODEL_DIR"), help="Directory to save the model.")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse of regularization strength.")
    args = parser.parse_args()
    train_model(args)