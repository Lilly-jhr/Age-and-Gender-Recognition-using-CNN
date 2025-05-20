import cv2
import numpy as np
import tensorflow as tf
import os

MODEL_PATH = 'models/age_gender_model.keras'

PROTOTXT_PATH = 'models/deploy.prototxt.txt'
CAFFEMODEL_PATH = 'models/res10_300x300_ssd_iter_140000.caffemodel'
DNN_CONFIDENCE_THRESHOLD = 0.5 

IMAGE_SIZE = (128, 128) 

loaded_model = None
face_net = None 

def load_resources():
    """Loads the Keras model and DNN face detector."""
    global loaded_model, face_net
    
    if loaded_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Keras model not found at {MODEL_PATH}")
        print(f"Loading Keras model from {MODEL_PATH}...")
        loaded_model = tf.keras.models.load_model(MODEL_PATH)
        print("Keras model loaded.")

    if face_net is None:
        if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(CAFFEMODEL_PATH):
            raise FileNotFoundError(f"DNN face detector files not found. Check PROTOTXT_PATH and CAFFEMODEL_PATH.")
        print(f"Loading DNN face detector from {PROTOTXT_PATH} and {CAFFEMODEL_PATH}...")
        face_net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
        if face_net.empty():
            raise IOError("Failed to load DNN face detector.")
        print("DNN face detector loaded.")

def detect_faces_dnn(image, current_face_net, confidence_threshold=DNN_CONFIDENCE_THRESHOLD):
    """
    Detects faces in an image using the DNN face detector.
    Args:
        image (numpy.ndarray): The input image (loaded by OpenCV - BGR).
        current_face_net (cv2.dnn_Net): The loaded DNN face detector.
        confidence_threshold (float): Minimum probability to filter weak detections.
    Returns:
        list: A list of (x, y, w, h) tuples for detected face bounding boxes.
    """
    if image is None:
        print("Error: Input image is None in detect_faces_dnn.")
        return []
    
    (h, w) = image.shape[:2]
    
    # Creates a blob from the image and perform a forward pass of the face detection model
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    current_face_net.setInput(blob)
    detections = current_face_net.forward()
    
    faces = []
    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections
        if confidence > confidence_threshold:
            # (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            face_w = endX - startX
            face_h = endY - startY

            if face_w > 0 and face_h > 0: 
                 faces.append((startX, startY, face_w, face_h))
    return faces


def preprocess_face(face_image):
    """
    Preprocesses a single cropped face image for model prediction.
    """
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    resized_face = cv2.resize(face_rgb, IMAGE_SIZE)
    normalized_face = resized_face.astype('float32') / 255.0
    preprocessed_face = np.expand_dims(normalized_face, axis=0)
    return preprocessed_face

def predict_on_image(image_or_path):
    """
    Performs face detection (DNN) and age/gender prediction on a single image.
    """
    global loaded_model, face_net
    if loaded_model is None or face_net is None:
        try:
            load_resources()
        except Exception as e:
            print(f"Error loading resources: {e}")
            if isinstance(image_or_path, str): img = cv2.imread(image_or_path)
            else: img = image_or_path.copy()
            return img if img is not None else np.zeros((100,100,3), dtype=np.uint8), []


    if isinstance(image_or_path, str):
        if not os.path.exists(image_or_path):
            print(f"Error: Image path does not exist: {image_or_path}")
            return np.zeros((100,100,3), dtype=np.uint8), []
        image = cv2.imread(image_or_path)
        if image is None:
            print(f"Error: Could not read image from path: {image_or_path}")
            return np.zeros((100,100,3), dtype=np.uint8), []
    elif isinstance(image_or_path, np.ndarray):
        image = image_or_path.copy()
    else:
        print("Error: Invalid input type for image_or_path.")
        return np.zeros((100,100,3), dtype=np.uint8), []

    original_image = image.copy()
    detected_faces = detect_faces_dnn(image, face_net, confidence_threshold=DNN_CONFIDENCE_THRESHOLD) 
    predictions_list = []

    if len(detected_faces) == 0:
        print("No faces detected.")
        return original_image, predictions_list

    for (x, y, w, h) in detected_faces:
        # DNN detector gives startX, startY, endX, endY.
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        
        cropped_face = image[y1:y2, x1:x2]

        if cropped_face.shape[0] == 0 or cropped_face.shape[1] == 0:
            print(f"Warning: Empty face crop at bbox ({x},{y},{w},{h}). Skipping.")
            continue

        preprocessed_face_for_model = preprocess_face(cropped_face)
        
        age_pred_raw, gender_pred_proba_raw = loaded_model.predict(preprocessed_face_for_model, verbose=0)
        
        predicted_age = int(round(age_pred_raw[0][0]))
        gender_probability = gender_pred_proba_raw[0][0]

        if gender_probability > 0.5:
            gender_label = "Male"
            confidence = gender_probability * 100
        else:
            gender_label = "Female"
            confidence = (1 - gender_probability) * 100
        
        predictions_list.append({
            'bbox': (x, y, w, h),
            'age': predicted_age,
            'gender': gender_label,
            'confidence': confidence
        })

    annotated_image = draw_predictions_on_image(original_image, predictions_list)
    return annotated_image, predictions_list



def draw_predictions_on_image(image_np, predictions):
    """
    Draws bounding boxes and predicted labels on the image.
    Args:
        image_np (numpy.ndarray): The original image.
        predictions (list): List of prediction dicts from predict_on_image.
    Returns:
        numpy.ndarray: The image with annotations.
    """
    display_image = image_np.copy()
    for pred in predictions:
        x, y, w, h = pred['bbox']
        age = pred['age']
        gender = pred['gender']
        confidence = pred['confidence']

        label = f"{gender} ({confidence:.1f}%) - Age: {age}"
        
        color = (0, 255, 0) 
        text_color = (0, 0, 0) 

        
        cv2.rectangle(display_image, (x, y), (x + w, y + h), color, 2)
        
        
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1     

        (label_width, label_height), baseline = cv2.getTextSize(label, font_face, font_scale, thickness)
        
        text_bg_y1 = max(y - label_height - baseline - 5, 0)
        text_bg_y2 = y - 5 
        
        if text_bg_y1 < 0 : 
             text_bg_y1 = y + baseline + 5 
             text_bg_y2 = y + label_height + baseline + 5

        cv2.rectangle(display_image, (x, text_bg_y1), 
                                     (x + label_width, text_bg_y2), 
                                     color, cv2.FILLED)
        
        text_y_pos = text_bg_y2 - baseline 

        cv2.putText(display_image, label, (x, text_y_pos), 
                    font_face, font_scale, text_color, thickness, cv2.LINE_AA)
                    
    return display_image

# for testing ---
if __name__ == '__main__':
    TEST_IMAGE_PATH = "data/test_img/kid 2.jpg" 
    if not os.path.exists(TEST_IMAGE_PATH) or TEST_IMAGE_PATH == "path_to_your_test_image_with_faces.jpg":
        print(f"Error: Test image path '{TEST_IMAGE_PATH}' is not set or does not exist.")
    else:
        try:
            load_resources() 
            
            annotated_img, preds = predict_on_image(TEST_IMAGE_PATH)

            if annotated_img is not None and annotated_img.size > 0 :
                print("\nPredictions:")
                for p in preds:
                    print(f"  Face at {p['bbox']}: Age={p['age']}, Gender={p['gender']} ({p['confidence']:.1f}%)")
                
                cv2.imshow("DNN Predictions", annotated_img)
                
                output_filename = "dnn_annotated_" + os.path.basename(TEST_IMAGE_PATH)
                cv2.imwrite(output_filename, annotated_img)
                print(f"Annotated image saved as {output_filename}")

                print("\nPress any key to close the image window...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("Inference failed or no image to show.")

        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()