
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm 

# config
IMAGE_SIZE = (128, 128)
DATASET_PATH = 'data/UTKFace' 

GENDER_MAP_MODEL_REQUIREMENT = {0: 1, 1: 0} 

def parse_filename(filename):
    """
    Parses the UTKFace filename to extract age, gender, and race.
    Example: 26_1_2_20170116185100484.jpg.chip.jpg
    """
    try:
        parts = filename.split('_')
        if len(parts) < 3:
            return None, None, None 

        age = int(parts[0])
        gender_original = int(parts[1]) 

        gender = GENDER_MAP_MODEL_REQUIREMENT.get(gender_original)
        if gender is None:
            print(f"Warning: Unexpected gender value '{gender_original}' in filename {filename}. Skipping.")
            return None, None, None

        # race = int(parts[2]) 
        return age, gender, None 
    except ValueError as e:
        print(f"Error parsing filename {filename}: {e}. Skipping.")
        return None, None, None
    except IndexError as e:
        print(f"Error parsing filename {filename} due to insufficient parts: {e}. Skipping.")
        return None, None, None

def load_and_preprocess_data(dataset_path=DATASET_PATH, image_size=IMAGE_SIZE):
    """
    Loads images from the dataset_path, preprocesses them, and extracts labels.
    """
    images = []
    ages = []
    genders = []
    
    filenames = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"Found {len(filenames)} images in {dataset_path}.")
    if not filenames:
        print(f"Error: No image files found in {dataset_path}. Please check the path and dataset.")
        return None, None, None

    for filename in tqdm(filenames, desc="Processing images"):
        age, gender, _ = parse_filename(filename)

        if age is None or gender is None:
            continue 

        # validate age and gender 
        if not (0 <= age <= 116):
            # print(f"Warning: Invalid age {age} in {filename}. Skipping.")
            continue
        if gender not in [0, 1]: 
            # print(f"Warning: Invalid gender {gender} (after mapping) in {filename}. Skipping.")
            continue
            
        img_path = os.path.join(dataset_path, filename)
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}. Skipping.")
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB
            img = cv2.resize(img, image_size)
            img = img.astype('float32') / 255.0 # normalize pixel values to [0, 1]
            
            images.append(img)
            ages.append(age)
            genders.append(gender)
            
        except Exception as e:
            print(f"Error processing image {img_path}: {e}. Skipping.")
            continue
            
    if not images:
        print("Error: No images were successfully processed. Check dataset and parsing logic.")
        return None, None, None

    print(f"Successfully processed {len(images)} images.")
    return np.array(images), np.array(ages), np.array(genders)

def get_data_splits(images, ages, genders, test_size=0.2, val_size=0.2, random_state=42):
    """
    Splits the data into training, validation, and testing sets.
    val_size is proportion of the (1-test_size) data.
    """
    if images is None or len(images) == 0:
        print("Error: Cannot split data as no images were loaded.")
        return (None,) * 6 

    # First split: Training + Validation vs. Test
    X_train_val, X_test, y_age_train_val, y_age_test, y_gender_train_val, y_gender_test = train_test_split(
        images, ages, genders, test_size=test_size, random_state=random_state, stratify=genders 
    )

    # Second split: Training vs. Validation
    # Adjust val_size relative to the new dataset size (X_train_val)
    relative_val_size = val_size / (1 - test_size)
    if len(np.unique(y_gender_train_val)) < 2: # Check if stratification is possible
        print("Warning: Not enough classes in y_gender_train_val for stratified split. Performing non-stratified split for validation.")
        X_train, X_val, y_age_train, y_age_val, y_gender_train, y_gender_val = train_test_split(
            X_train_val, y_age_train_val, y_gender_train_val, test_size=relative_val_size, random_state=random_state
        )
    else:
        X_train, X_val, y_age_train, y_age_val, y_gender_train, y_gender_val = train_test_split(
            X_train_val, y_age_train_val, y_gender_train_val, test_size=relative_val_size, random_state=random_state, stratify=y_gender_train_val
        )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    return X_train, X_val, X_test, \
           y_age_train, y_age_val, y_age_test, \
           y_gender_train, y_gender_val, y_gender_test

# for testing 
if __name__ == '__main__':
    print("Starting data preprocessing...")
    
    if not os.path.isdir(DATASET_PATH):
        print(f"Error: Dataset path '{DATASET_PATH}' does not exist or is not a directory.")
        print("Please ensure you have downloaded and extracted the UTKFace dataset into the 'data/UTKFace' folder.")
    else:
        images, ages, genders = load_and_preprocess_data()

        if images is not None and len(images) > 0:
            print(f"\nData loaded: {len(images)} images, {len(ages)} ages, {len(genders)} genders.")
            print(f"Image shape: {images.shape}")
            print(f"Ages min: {ages.min()}, max: {ages.max()}, example: {ages[:5]}")
            print(f"Genders unique: {np.unique(genders, return_counts=True)}, example: {genders[:5]}") 


            # Splitting data
            X_train, X_val, X_test, \
            y_age_train, y_age_val, y_age_test, \
            y_gender_train, y_gender_val, y_gender_test = get_data_splits(images, ages, genders)

            if X_train is not None:
                print(f"\nData splits successful.")
                print(f"X_train shape: {X_train.shape}, y_age_train shape: {y_age_train.shape}, y_gender_train shape: {y_gender_train.shape}")
                print(f"X_val shape: {X_val.shape}, y_age_val shape: {y_age_val.shape}, y_gender_val shape: {y_gender_val.shape}")
                print(f"X_test shape: {X_test.shape}, y_age_test shape: {y_age_test.shape}, y_gender_test shape: {y_gender_test.shape}")

                # Check for NaN or inf values in labels 
                print(f"NaN in y_age_train: {np.isnan(y_age_train).any()}")
                print(f"NaN in y_gender_train: {np.isnan(y_gender_train).any()}")

                # Save preprocessed data
                np.savez_compressed('data/preprocessed_data.npz',
                                    X_train=X_train, y_age_train=y_age_train, y_gender_train=y_gender_train,
                                     X_val=X_val, y_age_val=y_age_val, y_gender_val=y_gender_val,
                                     X_test=X_test, y_age_test=y_age_test, y_gender_test=y_gender_test)
                print("\nPreprocessed data saved to data/preprocessed_data.npz")
            else:
                print("\nData splitting failed.")
        else:
            print("\nData loading and preprocessing failed. Check logs for errors.")