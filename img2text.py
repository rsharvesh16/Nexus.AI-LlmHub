import streamlit as st
import google.generativeai as genai
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

def image2text_model():
    st.title("Image2Text Model")

    def get_gemini_response(input, image):
        model = genai.GenerativeModel('gemini-pro-vision')
        if input:
            response = model.generate_content([input, image])
        else:
            response = model.generate_content(image)
        return response.text

    input = st.text_input("Input Prompt: ", key="input")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image = ""

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
    submit = st.button("Generate Response")

    if submit:
        response = get_gemini_response(input, image)
        st.subheader("The Response is")
        st.write(response)
