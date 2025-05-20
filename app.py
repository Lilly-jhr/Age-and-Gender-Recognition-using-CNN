import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd

from src.inference import predict_on_image, load_resources as load_inference_resources

st.set_page_config(
    page_title="AI Face Analyzer",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

LOGO_PATH = "assets/logo.png" 

# Model and Resource Loading (Cached) 
@st.cache_resource
def load_all_resources_once():
    progress_bar = st.progress(0, text="Loading AI models, please wait...")
    try:
        progress_bar.progress(30, text="Loading face detection model...")
        load_inference_resources() 
        progress_bar.progress(100, text="All models loaded successfully!")
        st.session_state.models_loaded_successfully = True
        progress_bar.empty() 
    except FileNotFoundError as e:
        st.session_state.models_loaded_successfully = False
        st.error(f"😞 Model Loading Error: {e}. Ensure model files are in 'models/'.")
        st.info("Required: age_gender_model.keras, deploy.prototxt.txt, res10_300x300_ssd_iter_140000.caffemodel")
        progress_bar.empty()
    except Exception as e:
        st.session_state.models_loaded_successfully = False
        st.error(f"🤯 An unexpected error occurred during model loading: {e}")
        progress_bar.empty()

if 'models_loaded_successfully' not in st.session_state:
    load_all_resources_once()



def main_app():
    col_title_1, col_title_2, col_title_3 = st.columns([0.5, 3, 0.5])
    with col_title_2:
        if os.path.exists(LOGO_PATH):
            logo_image = Image.open(LOGO_PATH)
            st.image(logo_image, width=120) 
        st.title("✨ AI Face Analyzer ✨")
        st.caption("Instant Age & Gender Insights from Your Images")

    st.markdown("---")

    if not st.session_state.get('models_loaded_successfully', False):
        st.warning("🚦 Critical: AI Models could not be loaded. The application cannot proceed. Please check for error messages above and ensure model files are correctly placed.")
        st.stop()

    st.markdown(
        "👋 Welcome! Upload an image, and our AI will detect faces and estimate their age and gender."
    )

    uploaded_file = st.file_uploader(
        "🖼️ **Upload Your Image Here:**", 
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG. Max file size: 200MB.",
        label_visibility="visible"
    )
    
    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file).convert('RGB')
            opencv_image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            st.markdown("---")
            st.subheader("🔬 Analysis Results")

            with st.spinner("🧠 Thinking... Analyzing faces..."):
                annotated_image_bgr, predictions = predict_on_image(opencv_image_bgr)

            if annotated_image_bgr is None:
                st.error("❌ An internal error occurred during image processing.")
                return

            annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)
            
            if not predictions:
                # If no faces are detected, show the original uploaded image
                st.image(pil_image, caption="Uploaded Image - No Faces Detected", use_column_width='auto') 
                st.info("🤔 No faces were detected in the uploaded image. Try another one perhaps?")
                return

            col_img, col_data = st.columns([0.6, 0.4]) 

            with col_img:
                # Display the image with annotations
                st.image(annotated_image_rgb, caption="Processed Image with Predictions", use_column_width='always') 
            
            with col_data:
                st.markdown("#### 🧑‍🤝‍🧑 Detected Individuals")
                df_data = []
                for i, pred in enumerate(predictions):
                    df_data.append({
                        "👤 ID": i + 1,
                        "🎂 Age (Est.)": pred['age'],
                        "🚻 Gender (Est.)": pred['gender'],
                        "✅ Confidence": f"{pred['confidence']:.1f}%"
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(
                    df, 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        "👤 ID": st.column_config.NumberColumn("ID", width="small", help="Unique ID for detected face"),
                        "🎂 Age (Est.)": st.column_config.NumberColumn("Age", help="Estimated age in years"),
                        "🚻 Gender (Est.)": st.column_config.TextColumn("Gender", help="Estimated gender"),
                        "✅ Confidence": st.column_config.TextColumn("Gender Conf.", help="Confidence score for gender prediction"),
                    }
                )
        
        except Exception as e:
            st.error(f"💥 Oops! Something went wrong: {str(e)}")
            st.caption("If the problem persists, please try a different image or check the image format.")
    else:
        if st.session_state.get('models_loaded_successfully', False):
            st.info("☝️ **Ready when you are!** Upload an image to begin.")
    
    st.markdown("---")
    
    # footer
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; color: #777; font-size: 0.9em;">
        <p>Developed with ❤️ by <strong> BOLD </strong></p>
        <p>Powered by: OpenCV | TensorFlow/Keras | Streamlit</p>
        <p><a href="https://github.com/Lilly-jhr/Age-and-Gender-Recognition-using-CNN.git" target="_blank" style="color: #22A7F0; text-decoration: none;">View on GitHub 🐙</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    if not os.path.exists("assets"):
        os.makedirs("assets")
    main_app()