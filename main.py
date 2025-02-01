import argparse
from pipelines.train_pipeline import trainer
from pipelines.evaluate_pipeline import evaluator

# Define argument parser
def main():
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection Pipeline")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    args = parser.parse_args()
    
    if args.train:
        print("Starting training pipeline...")
        trainer()
    
    if args.evaluate:
        print("Starting evaluation pipeline...")
        evaluator()

if __name__ == "__main__":
    main()
