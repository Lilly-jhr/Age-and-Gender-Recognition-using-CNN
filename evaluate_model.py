
import os
import numpy as np
import tensorflow as tf



PREPROCESSED_DATA_PATH = 'data/preprocessed_data.npz'
SAVED_MODEL_PATH = 'models/age_gender_model.keras' 

def load_test_data(path):
    """Loads only the test set from the preprocessed .npz file."""
    print(f"Loading test data from {path}...")
    if not os.path.exists(path):
        print(f"Error: Preprocessed data file not found at {path}.")
        print("Please ensure 'data/preprocessed_data.npz' exists.")
        return None
    
    try:
        data = np.load(path)
        X_test = data['X_test']
        y_age_test = data['y_age_test']
        y_gender_test = data['y_gender_test']
        print("Test data loaded successfully.")
        return X_test, y_age_test, y_gender_test
    except KeyError as e:
        print(f"Error: Missing expected key in data file {path}: {e}")
        print("Ensure X_test, y_age_test, y_gender_test were saved by data_preprocessing.py")
        return None
    except Exception as e:
        print(f"Error loading test data from {path}: {e}")
        return None

def main():
    # 1. Load test data
    load_result = load_test_data(PREPROCESSED_DATA_PATH)
    if load_result is None:
        return
    X_test, y_age_test, y_gender_test = load_result

    print(f"X_test shape: {X_test.shape}, y_age_test shape: {y_age_test.shape}, y_gender_test shape: {y_gender_test.shape}")

    # 2. Load the trained model
    if not os.path.exists(SAVED_MODEL_PATH):
        print(f"Error: Trained model not found at {SAVED_MODEL_PATH}")
        print("Please train the model first using train_model.py")
        return
    
    print(f"Loading trained model from {SAVED_MODEL_PATH}...")
    try:
        
        model = tf.keras.models.load_model(SAVED_MODEL_PATH)
        print("Model loaded successfully.")
        model.summary(line_length=120)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Prepare target data for evaluation
    y_test_dict = {'age_output': y_age_test, 'gender_output': y_gender_test}

    # 4. Evaluate the model on the test set
    print("\n--- Evaluating on Test Set ---")
    eval_results_test = model.evaluate(X_test, y_test_dict, verbose=1, batch_size=32)

    # 5. Print the test metrics
    print(f"\nModel metrics_names after loading: {model.metrics_names}") 
    print(f"Raw evaluation results: {eval_results_test}") 
    
    if len(eval_results_test) == 5: 
        test_loss = eval_results_test[0]
        test_age_loss = eval_results_test[1]       
        test_gender_loss = eval_results_test[2]    
        test_age_mae = eval_results_test[3]        
        test_gender_accuracy = eval_results_test[4]

        print(f"\nTest Set Performance:")
        print(f"Overall Test Loss: {test_loss:.4f}")
        print(f"Test Age Loss (MSE): {test_age_loss:.4f}")
        print(f"Test Gender Loss (BCE): {test_gender_loss:.4f}")
        print(f"Test Age MAE: {test_age_mae:.2f} years")
        print(f"Test Gender Accuracy: {test_gender_accuracy*100:.2f}%")
    else:
        print("Error: Unexpected number of results from model.evaluate().")
        print(f"Expected 5, got {len(eval_results_test)}.")
        print(f"Raw results: {eval_results_test}")
        print(f"Model metrics names: {model.metrics_names}")



if __name__ == '__main__':
    main()