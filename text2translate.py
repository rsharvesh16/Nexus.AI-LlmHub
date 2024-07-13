import streamlit as st
import google.generativeai as genai
import os

def text2translate_model():
    st.title("Text2Translate Model")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-pro")

    def generate_response(input_text, prompt):
        response = model.generate_content([input_text, "", prompt])
        return response.text

    def main():
        st.title("Text-2-Translation Model")
        st.header("Google Gemini AI Model")
        input_text = st.text_input("Enter your text:", key="input")
        input_lang = st.text_input("Enter your language to Translate", key=input)
        prompt = f"Translate the exact following text to {input_lang}: {input_text}"

        if st.button("Generate Output"):
            response = generate_response(prompt, input_text)
            st.subheader("The Response Is")
            st.write(response)
    main()
