# Age and Gender Prediction CNN 🖼️✨

This project implements a complete deep learning pipeline for real-time, multi-face age and gender prediction using a custom Convolutional Neural Network (CNN) model, deployed as an intuitive Streamlit web application.


## 🌟 Features

*   **Multi-Face Detection:** Utilizes OpenCV's DNN module (Caffe model) to accurately detect multiple faces in an image.
*   **Age Prediction:** A custom CNN model regresses the age of each detected face.
*   **Gender Prediction:** The same CNN model classifies the gender (Male/Female) of each detected face, along with a confidence score.
*   **Interactive Web Application:** Built with Streamlit for easy image uploads and visualization of results.
*   **Real-Time (Potential):** Optimized for efficient inference, with potential for real-time video processing (not implemented in this version).
*   **Modular Code:** Organized structure for data preprocessing, model building, training, inference, and the application.

## 🛠️ Tech Stack

*   **Python 3.x**
*   **TensorFlow & Keras:** For building and training the custom CNN model.
*   **OpenCV:** For image processing and face detection (DNN module).
*   **Streamlit:** For creating the interactive web application.
*   **NumPy & Pandas:** For numerical operations and data handling.
*   **Scikit-learn:** For utility functions like class weight calculation.
*   **Dataset:** UTKFace (or a similar dataset with age/gender labels).

## 📸 Example Output

Here's an example of the model predicting age and gender for multiple faces:

![Example Output 1](example_outputs/dnn_annotated_family.jpg) <!-- Replace with one of your good output images -->
*(Caption: Predictions on a family photo.)*

<!-- Add another example if you like -->
<!-- ![Example Output 2](example_outputs/dnn_annotated_another.jpg) -->


## 📁 Project Structure
Use code with caution.
Markdown
age-gender-prediction/
├── assets/
├── data/
│ ├── UTKFace/ # Dataset (Not committed to Git - Download separately)
│ └── preprocessed_data.npz # (Not committed - Generated by script)
├── example_outputs/ # Example annotated images
│ └── dnn_annotated_family.jpg
├── models/ # Trained models and detector files
│ ├── age_gender_model.keras # Our trained Keras model
│ ├── haarcascade_frontalface_default.xml # (Alternative Haar detector)
│ ├── deploy.prototxt.txt # DNN Face Detector config
│ └── res10_300x300_ssd_iter_140000.caffemodel # DNN Face Detector weights
│ └── model_architecture_v2.png # Model architecture diagram
├── plots/ # Training history plots 
├── src/ # Source code
│ ├── init.py
│ ├── data_preprocessing.py
│ ├── model_builder.py
│ ├── inference.py
│ └── utils.py 
├── .gitignore
├── app.py # Streamlit application
├── evaluate_model.py # Script for model evaluation
├── requirements.txt # Python dependencies
├── train_model.py # Script for model training
└── README.md # This file

## 🚀 Getting Started

### Prerequisites

*   Python 3.8 or higher
*   pip (Python package installer)
*   Git (for cloning the repository)
*   Graphviz (Optional, only if you want to regenerate the model architecture diagram using `src/model_builder.py`. See [Graphviz Downloads](https://graphviz.org/download/)).

### Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/[YourGitHubUsername]/[YourRepositoryName].git
    cd [YourRepositoryName]
    ```
    *(Replace `[YourGitHubUsername]/[YourRepositoryName]` with your actual repo path)*

2.  **Create and Activate a Virtual Environment:**
    (Recommended to avoid conflicts with other Python projects)
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Dataset (UTKFace):**
    *   Download the UTKFace dataset. A common source is Kaggle or by searching "UTKFace dataset download".
    *   Extract the dataset.
    *   Ensure all JPEG image files are placed directly under the `data/UTKFace/` directory within the project structure. (e.g., `age-gender-prediction/data/UTKFace/100_0_0_20170112213500903.jpg.chip.jpg`).
    *   *Note: The dataset is not included in this repository due to its size.*

5.  **Download Face Detector Models:**
    The OpenCV DNN face detector model files (`deploy.prototxt.txt` and `res10_300x300_ssd_iter_140000.caffemodel`) are included in the `models/` directory. If they are missing or you prefer to download them manually:
    *   `deploy.prototxt.txt`: [OpenCV GitHub](https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt.txt)
    *   `res10_300x300_ssd_iter_140000.caffemodel`: [OpenCV 3rd Party GitHub](https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)
    Place these files in the `models/` directory.

6.  **Obtain/Generate Trained Model (`age_gender_model.keras`):**
    *   **Option A (Use Pre-trained - If Provided):** If a `age_gender_model.keras` file is provided in the `models/` directory of this repository, you can use it directly.
    *   **Option B (Train Your Own):**
        1.  **Preprocess Data:** Run the data preprocessing script. This will generate `data/preprocessed_data.npz`.
            ```bash
            python src/data_preprocessing.py
            ```
        2.  **Train the Model:** Run the training script. This will save the best model as `models/age_gender_model.keras`.
            ```bash
            python train_model.py
            ```
            *(Note: Training can take several hours depending on your hardware and dataset size.)*

### Running the Application

Once all dependencies are installed and the necessary model files (`age_gender_model.keras`, `deploy.prototxt.txt`, `res10_300x300_ssd_iter_140000.caffemodel`) are in the `models/` directory:

1.  **Launch the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
2.  Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).
3.  Upload an image and see the predictions!

## 📈 Model Performance

The model was trained on a subset of the UTKFace dataset (approx.14,224images for training, 4,742 for validation, 4742 for testing).
The performance on the test set is as follows:

*   **Age Prediction (Mean Absolute Error - MAE):** ~5.81 years
*   **Gender Prediction (Accuracy):** ~82.35%

*(Note: These metrics are based on the specific training run and dataset subset used. Results may vary.)*

## 🔮 Future Work / Potential Improvements

*   **Improve Model Accuracy:**
    *   Train on the full UTKFace dataset or even larger datasets.
    *   Experiment with different CNN architectures or pre-trained backbones (Transfer Learning).
    *   Advanced hyperparameter tuning.
    *   More sophisticated data augmentation techniques.
    *   Implement robust class weighting for gender if imbalance is significant in larger datasets.
*   **Advanced Face Detection:** Explore MTCNN or other state-of-the-art face detectors for even better detection and alignment.
*   **Real-time Video Processing:** Adapt the pipeline to process video streams from a webcam.
*   **Deployment Enhancements:**
    *   Containerize with Docker for easier deployment.
    *   Optimize model size (e.g., using TensorFlow Lite) for edge devices or faster loading.
*   **Additional Attributes:** Extend the model to predict ethnicity or emotion.

## 📜 License

This project is licensed under the MIT License - see the `LICENSE` file for details (if you choose to add one).

## 🙏 Acknowledgements

*   The creators of the UTKFace dataset.
*   The developers of TensorFlow, Keras, OpenCV, and Streamlit.

---

Developed by **BOLD** - [Link to your GitHub Profile or Website, Optional]
