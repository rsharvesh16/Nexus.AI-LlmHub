import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai 
import os
from PIL import Image
from io import BytesIO
import base64
import boto3
import json
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Function for each model
def text2text_model():
    st.title("Text2Text Model")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-pro")

    def generate_response(input_text, prompt):
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content([input_text, "", prompt])
        return response.text

    def main():
        st.header("Google Gemini Pro AI Model")

        # Get user input
        input_text = st.text_input("Enter your text:", key="input")

        prompt = """You are an expert in understanding the given question and you have a clear mind.
        When a user gives a query give them accurate answers and explain them in detail with points,
        Answer them accurately and correctly and dont give wrong answers. You are an intellingent in
        doing this. Avoid speaking contents related to abuse, religion, sexual abuse etc which may
        give a conflict to the user. Be Polite and Generous."""

        if st.button("Generate Output"):
            # Generate and display the output
            response = generate_response(prompt, input_text)
            st.subheader("The Response Is")
            st.write(response)
    main()

def text2image_model():
    st.title("Text2Image Model")
    bedrock_runtime = boto3.client(service_name="bedrock-runtime")
    def get_amazon_llm(text_input):
        body = json.dumps(
            {
                "taskType": "TEXT_IMAGE",
                "textToImageParams": {
                    "text": text_input, 
                },
                "imageGenerationConfig": {
                    "numberOfImages": 1,   # Range: 1 to 5 
                    "quality": "premium",  # Options: standard or premium
                    "height": 768,         # Supported height list in the docs 
                    "width": 1280,         # Supported width list in the docs
                    "cfgScale": 7.5,       # Range: 1.0 (exclusive) to 10.0
                    "seed": 42             # Range: 0 to 214783647
                }
            }
        )
        
        try:
            response = bedrock_runtime.invoke_model(
                body=body, 
                modelId="amazon.titan-image-generator-v1",
                accept="application/json", 
                contentType="application/json"
            )
            
            # Read the StreamingBody and decode it to a string
            response_body = response['body'].read().decode('utf-8')
            
            return response_body
        except bedrock_runtime.exceptions.ValidationException as e:
            st.error("The prompt was flagged by content filters. Please adjust your prompt and try again.")
            return None
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None
    def main():

        st.header("Image Generator AWS Bedrock💁")

        text_input = st.text_input("Ask a prompt to generate an image")
        
        if st.button("Generate Image"):
            if text_input:
                response = get_amazon_llm(text_input)
                if response:
                    try:
                        response_json = json.loads(response)
                        st.json(response_json)  # Print the JSON response for debugging
                        
                        # Check if the response contains a list of images
                        images_data = response_json.get('images')
                        if images_data and isinstance(images_data, list):
                            for image_data in images_data:
                                image = Image.open(BytesIO(base64.b64decode(image_data)))
                                st.image(image, caption="Generated Image")
                        else:
                            st.error("No image data found in the response.")
                    except json.JSONDecodeError:
                        st.error("Failed to decode JSON response.")
                    # except KeyError as e:
                    #     st.error(f"Key error: {e}")
                    # except Exception as e:
                    #     st.error(f"An error occurred while processing the response: {e}")
                else:
                    st.error("Failed to generate image. Please try again with a different prompt.")
    main()

def text2code_model():
    st.title("Text2Code Model")
    bedrock = boto3.client(service_name="bedrock-runtime")
    def get_mistral_llm():
        llm = Bedrock(model_id="mistral.mistral-7b-instruct-v0:2", client=bedrock)
        return llm
    prompt_template = """
    Human: Use the following pieces of context to convert the text into code. Provide a concise solution but use at least 250 words to summarize with detailed explanations. If you don't know how to convert the text into code, just say that you don't know, don't try to make up a solution.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    def get_response_llm(llm, context, question):
        chain = LLMChain(llm=llm, prompt=PROMPT)
        answer = chain.run(context=context, question=question)
        return answer
    
    def main():
        st.header("Text 2 Code using AWS Bedrock💁")

        context = st.text_area("Ask a Question")
        user_question = st.header("Response:")
        if st.button("Mistral Output"):
            llm = get_mistral_llm()
            st.write(get_response_llm(llm, context, user_question))
    main()

def text2translate_model():
    st.title("Text2Translate Model")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-pro") 

    def main():
        st.title("Text-2-Translation Model")

        st.header("Google Gemini AI Model")

        # Get user input
        input_text = st.text_input("Enter your text:", key="input")
        input_lang = st.text_input("Enter your language to Translate", key=input)

        prompt = f"Translate the exact following text to {input_lang}: {input_text}"
        

        if st.button("Generate Output"):
            # Generate and display the output
            response = generate_response(prompt, input_text)
            st.subheader("The Response Is")
            st.write(response)
    main()

def generate_response(input_text, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content([input_text, "", prompt])
    return response.text

def image2text_model():

    st.title("Image2Text Model")

    def get_gemini_response(input,image):
        model = genai.GenerativeModel('gemini-pro-vision')
        if input!="":
            response = model.generate_content([input,image])
        else:
            response = model.generate_content(image)
        return response.text
    
    input=st.text_input("Input Prompt: ",key="input")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image=""  

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
    submit=st.button("Generate Response")

    if submit:
        response=get_gemini_response(input,image)
        st.subheader("The Response is")
        st.write(response)

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
