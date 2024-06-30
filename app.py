import streamlit as st

# Function for each model
def text2text_model():
    st.title("Text2Text Model")
    st.write("This is the content for the Text2Text Model.")

def text2image_model():
    st.title("Text2Image Model")
    st.write("This is the content for the Text2Image Model.")

def text2code_model():
    st.title("Text2Code Model")
    st.write("This is the content for the Text2Code Model.")

def text2translate_model():
    st.title("Text2Translate Model")
    st.write("This is the content for the Text2Translate Model.")

def image2text_model():
    st.title("Image2Text Model")
    st.write("This is the content for the Image2Text Model.")

# Sidebar menu
st.sidebar.title("Menu")
menu = st.sidebar.radio("Select a Model", 
                        ["Text2Text Model", 
                         "Text2Image Model", 
                         "Text2Code Model", 
                         "Text2Translate Model", 
                         "Image2Text Model"])

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
