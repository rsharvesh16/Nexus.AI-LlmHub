import streamlit as st
import boto3
import json
from PIL import Image
from io import BytesIO
import base64

def text2image_model():
    st.title("Text2Image Model")
    bedrock_runtime = boto3.client(service_name="bedrock-runtime")

    def get_amazon_llm(text_input):
        body = json.dumps({
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {"text": text_input},
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "quality": "premium",
                "height": 768,
                "width": 1280,
                "cfgScale": 7.5,
                "seed": 42
            }
        })
        try:
            response = bedrock_runtime.invoke_model(
                body=body,
                modelId="amazon.titan-image-generator-v1",
                accept="application/json",
                contentType="application/json"
            )
            response_body = response['body'].read().decode('utf-8')
            return response_body
        except bedrock_runtime.exceptions.ValidationException:
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
                        st.json(response_json)
                        images_data = response_json.get('images')
                        if images_data and isinstance(images_data, list):
                            for image_data in images_data:
                                image = Image.open(BytesIO(base64.b64decode(image_data)))
                                st.image(image, caption="Generated Image")
                        else:
                            st.error("No image data found in the response.")
                    except json.JSONDecodeError:
                        st.error("Failed to decode JSON response.")
    main()
