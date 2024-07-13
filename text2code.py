import streamlit as st
import boto3
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

load_dotenv()

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

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

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
