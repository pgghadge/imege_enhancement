import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tempfile
import os

# Function to perform image enhancement
def enhance_image(image, enhancement_type):
    if enhancement_type == "Contrast Adjustment":
        contrast_factor = 1.5
        return cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)
    elif enhancement_type == "Brightness Adjustment":
        brightness_value = 50
        return cv2.convertScaleAbs(image, alpha=1, beta=brightness_value)
    elif enhancement_type == "Smoothing (Blurring)":
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif enhancement_type == "Image Sharpening":
        blurred = cv2.GaussianBlur(image, (5, 5), 10)
        return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    elif enhancement_type == "Masking":
        mask = np.zeros_like(image)
        height, width, _ = image.shape
        center = (width // 2, height // 2)
        radius = min(width, height) // 3
        cv2.circle(mask, center, radius, (255, 255, 255), thickness=cv2.FILLED)
        return cv2.bitwise_and(image, mask)
    elif enhancement_type == "Morphological Erosion":
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(image, kernel, iterations=1)
    elif enhancement_type == "Morphological Dilation":
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

# Streamlit app title
st.title("Image Enhancement App")

# Sidebar with enhancement options
st.sidebar.title("Enhancement Options")
enhancement_type = st.sidebar.selectbox("Select Enhancement Type", ["Contrast Adjustment", "Brightness Adjustment", "Smoothing (Blurring)", "Image Sharpening", "Masking", "Morphological Erosion", "Morphological Dilation"])

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Read and display the uploaded image
    image = Image.open(uploaded_image)

    # Create columns for original and enhanced images
    col1, col2 = st.columns(2)

    # Display the original image in the left column
    col1.header("Original Image")
    col1.image(image, use_column_width=True)

    # Perform image enhancement
    enhanced_image = enhance_image(np.array(image), enhancement_type)
    # enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)

    # Display the enhanced image in the right column with empty lines for vertical space
    col2.header(f"{enhancement_type} Enhanced Image")  # Title
    col2.image(enhanced_image, use_column_width=True)

    # Save enhanced image as a temporary PNG file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_filename = temp_file.name
        cv2.imwrite(temp_filename, enhanced_image)

    # Download enhanced image
    st.markdown("### Download Enhanced Image")
    st.write("Click the link below to download the enhanced image.")
    st.download_button("Download Enhanced Image", temp_filename, f"{enhancement_type.replace(' ', '_').lower()}_enhanced_image.png")

    # Clean up temporary file after download
    os.remove(temp_filename)
