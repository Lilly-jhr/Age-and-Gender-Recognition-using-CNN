import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.utils import class_weight


from src.model_builder import build_age_gender_model 
from src.data_preprocessing import IMAGE_SIZE 

# config
PREPROCESSED_DATA_PATH = 'data/preprocessed_data.npz'
MODEL_SAVE_PATH = 'models/age_gender_model.keras' 
LOG_DIR = 'logs/fit' 
PLOTS_DIR = 'plots' 

# Training Hyperparameters
BATCH_SIZE = 32
EPOCHS = 100 
INITIAL_LEARNING_RATE = 0.0005 
L2_REG_FACTOR = 1e-5       

def load_preprocessed_data(path):
    """Loads preprocessed data from .npz file."""
    print(f"Loading preprocessed data from {path}...")
    if not os.path.exists(path):
        print(f"Error: Preprocessed data file not found at {path}.")
        return None
    
    try:
        data = np.load(path)
        X_train = data['X_train']
        y_age_train = data['y_age_train']
        y_gender_train = data['y_gender_train']
        X_val = data['X_val']
        y_age_val = data['y_age_val']
        y_gender_val = data['y_gender_val']
        # X_test, y_age_test, y_gender_test 

        print("Data loaded successfully.")
        return (X_train, y_age_train, y_gender_train,
                X_val, y_age_val, y_gender_val)
    except Exception as e:
        print(f"Error loading data from {path}: {e}")
        return None

def plot_training_history(history, save_dir):
    """Plots and saves training history for age and gender."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Age training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['age_output_loss'], label='Age Train Loss (MSE)')
    plt.plot(history.history['val_age_output_loss'], label='Age Val Loss (MSE)')
    plt.title('Age Model Loss (MSE)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['age_output_mae'], label='Age Train MAE')
    plt.plot(history.history['val_age_output_mae'], label='Age Val MAE')
    plt.title('Age Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'age_training_history.png'))
    print(f"Age training history plot saved to {os.path.join(save_dir, 'age_training_history.png')}")
    plt.close()

    # Gender training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['gender_output_loss'], label='Gender Train Loss (BCE)')
    plt.plot(history.history['val_gender_output_loss'], label='Gender Val Loss (BCE)')
    plt.title('Gender Model Loss (Binary Crossentropy)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['gender_output_accuracy'], label='Gender Train Accuracy')
    plt.plot(history.history['val_gender_output_accuracy'], label='Gender Val Accuracy')
    plt.title('Gender Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gender_training_history.png'))
    print(f"Gender training history plot saved to {os.path.join(save_dir, 'gender_training_history.png')}")
    plt.close()


def main():
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1. Load preprocessed data
    data_load_result = load_preprocessed_data(PREPROCESSED_DATA_PATH)
    if data_load_result is None:
        return
    X_train, y_age_train, y_gender_train, X_val, y_age_val, y_gender_val = data_load_result

    print(f"X_train shape: {X_train.shape}, y_age_train shape: {y_age_train.shape}, y_gender_train shape: {y_gender_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_age_val shape: {y_age_val.shape}, y_gender_val shape: {y_gender_val.shape}")

    # 2. Calculate class weights for gender (0=female, 1=male)
    gender_classes = np.unique(y_gender_train)
    gender_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=gender_classes,
        y=y_gender_train
    )
    gender_class_weights_dict = dict(zip(gender_classes, gender_weights_array))
    print(f"Calculated gender class weights: {gender_class_weights_dict}")


    # 3. Build the model
    input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    model = build_age_gender_model(
        input_shape=input_shape,
        l2_reg=L2_REG_FACTOR,
        optimizer_lr=INITIAL_LEARNING_RATE
    )
    model.summary(line_length=120)

    # 4. Define Callbacks (same as before)
    checkpoint_cb = ModelCheckpoint(
        filepath=MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True,
        save_weights_only=False, mode='min', verbose=1
    )
    early_stopping_cb = EarlyStopping(
        monitor='val_loss', patience=15, verbose=1, mode='min', restore_best_weights=True
    )
    reduce_lr_cb = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1, mode='min'
    )
    tensorboard_cb = TensorBoard(
        log_dir=LOG_DIR, histogram_freq=1, write_graph=True, write_images=False, update_freq='epoch'
    )
    callbacks_list = [checkpoint_cb, early_stopping_cb, reduce_lr_cb, tensorboard_cb]

    # 5. Train the model
    print("\n--- Starting Model Training ---")
    y_train_dict = {'age_output': y_age_train, 'gender_output': y_gender_train}
    y_val_dict = {'age_output': y_age_val, 'gender_output': y_gender_val}

    
    # Create sample weights for the gender output for the training set
    gender_sample_weights_train = np.array([gender_class_weights_dict[g] for g in y_gender_train.astype(int)]) 

    age_sample_weights_train = np.ones_like(y_age_train, dtype=float)

    fit_sample_weights = {
        'age_output': age_sample_weights_train,
        'gender_output': gender_sample_weights_train
    }


    history = model.fit(
        X_train,
        y_train_dict,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val_dict),
        callbacks=callbacks_list,
        #sample_weight=fit_sample_weights, 
        verbose=1
    )

    print("\n--- Model Training Finished ---")

    # 6. Evaluate the best model
    print("\nEvaluating the best model on validation data (after training):")
    eval_results = model.evaluate(X_val, y_val_dict, verbose=0)
    
    val_loss = eval_results[0]
    val_age_loss = eval_results[1] 
    val_gender_loss = eval_results[2] 
    val_age_mae = eval_results[3] 
    val_gender_accuracy = eval_results[4]

    print(f"Overall Validation Loss: {val_loss:.4f}")
    print(f"Validation Age Loss (MSE): {val_age_loss:.4f}")
    print(f"Validation Gender Loss (BCE): {val_gender_loss:.4f}")
    print(f"Validation Age MAE: {val_age_mae:.2f} years")
    print(f"Validation Gender Accuracy: {val_gender_accuracy*100:.2f}%")

    # 7. Plot and save training history
    plot_training_history(history, PLOTS_DIR)

    print(f"\nBest model saved to: {MODEL_SAVE_PATH}")
    print(f"TensorBoard logs are in: {LOG_DIR}")
    print(f"To view TensorBoard, run: tensorboard --logdir \"{LOG_DIR}\"")

if __name__ == '__main__':
    main()