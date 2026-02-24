import streamlit as st
st.set_page_config(page_title="Digit Recognizer", layout="centered")

import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os

# ---------- Load Model ----------
#model_path = os.path.join(os.path.dirname(__file__), "mnist_cnn.keras")
model = load_model("mnist_cnn.h5")


st.title("🖊️ Handwritten Digit Recognizer (CNN)")
st.write("Draw a digit from 0 to 9 and click Predict")

# ---------- Canvas ----------
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# ---------- Prediction ----------
if st.button("Predict"):
    if canvas_result.image_data is not None:

        # Convert canvas to image
        img = Image.fromarray(
            (canvas_result.image_data[:, :, 0]).astype(np.uint8)
        )

        # Resize to MNIST size
        img = img.resize((28, 28))

        img_array = np.array(img)

        # Invert colors (VERY IMPORTANT)
        img_array = 255 - img_array

        # Normalize
        img_array = img_array / 255.0

        # Reshape for CNN
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        st.success(f"Predicted Digit: {digit}")

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os

# ---------- Load Model ----------
#model_path = os.path.join(os.path.dirname(__file__), "mnist_cnn.keras")
model = load_model("mnist_cnn.h5")
st.set_page_config(page_title="Digit Recognizer", layout="centered")

st.title("🖊️ Handwritten Digit Recognizer (CNN)")
st.write("Draw a digit from 0 to 9 and click Predict")

# ---------- Canvas ----------
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# ---------- Prediction ----------
if st.button("Predict"):
    if canvas_result.image_data is not None:

        # Convert canvas to image
        img = Image.fromarray(
            (canvas_result.image_data[:, :, 0]).astype(np.uint8)
        )

        # Resize to MNIST size
        img = img.resize((28, 28))

        img_array = np.array(img)

        # Invert colors (VERY IMPORTANT)
        img_array = 255 - img_array

        # Normalize
        img_array = img_array / 255.0

        # Reshape for CNN
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        st.success(f"Predicted Digit: {digit}")
        st.write(f"Confidence: {confidence:.2f}")
