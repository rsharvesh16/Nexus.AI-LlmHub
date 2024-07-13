import streamlit as st
from text2text import text2text_model
from text2img import text2image_model
from text2code import text2code_model
from text2translate import text2translate_model
from img2text import image2text_model
from pdf2text import pdf2text_model

# Sidebar menu
st.sidebar.title("Menu")
menu = st.sidebar.radio("Select a Model",
                        ["Text2Text Model",
                         "Text2Image Model",
                         "Text2Code Model",
                         "Text2Translate Model",
                         "Image2Text Model",
                         "PDF2Text (RAG)"])

# Main dashboard
if menu == "Text2Text Model":
    text2text_model()
elif menu == "Text2Image Model":
    text2image_model()
elif menu == "Text2Code Model":
    text2code_model()
elif menu == "Text2Translate Model":
    text2translate_model()
elif menu == "Image2Text Model":
    image2text_model()
elif menu == "PDF2Text (RAG)":
    pdf2text_model()