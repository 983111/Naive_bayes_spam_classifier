import os
import pickle
from model.preprocess import clean_text

def main():
    model_path = "spam_classifier.pkl"
    if not os.path.exists(model_path):
        print(f"Error: '{model_path}' not found. Please run 'python train.py' first.")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print("=== Spam Detector CLI ===")
    print("Type any text message to test, or type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("Enter message: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting.")
                break

            tokens = clean_text(user_input)
            prediction = model.predict_single(tokens)
            print(f"Prediction -> {prediction.upper()}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()