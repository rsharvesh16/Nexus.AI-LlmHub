import streamlit as st
import google.generativeai as genai
import os

def text2text_model():
    st.title("Text2Text Model")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-pro")

    def generate_response(input_text, prompt):
        response = model.generate_content([input_text, "", prompt])
        return response.text

    def main():
        st.header("Google Gemini Pro AI Model")

        input_text = st.text_input("Enter your text:", key="input")

        prompt = """You are an expert in understanding the given question and you have a clear mind.
        When a user gives a query give them accurate answers and explain them in detail with points,
        Answer them accurately and correctly and don't give wrong answers. You are intelligent in
        doing this. Avoid speaking contents related to abuse, religion, sexual abuse etc which may
        give a conflict to the user. Be Polite and Generous."""

        if st.button("Generate Output"):
            response = generate_response(prompt, input_text)
            st.subheader("The Response Is")
            st.write(response)
    main()
