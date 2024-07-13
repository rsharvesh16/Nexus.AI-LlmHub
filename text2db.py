import streamlit as st
import os
from sqlalchemy import create_engine
from llama_index.response.pprint_utils import pprint_response
from llama_index import SQLDatabase, ServiceContext
from llama_index.indices.struct_store.sql_query import NLSQLTableQueryEngine
import google.generativeai as genai

# Set up OpenAI LLM
llm = genai.GenerativeModel('gemini-pro')
service_context = ServiceContext.from_defaults(llm=llm)

# Define the folder to store uploaded databases
DATABASE_FOLDER = "database"

if not os.path.exists(DATABASE_FOLDER):
    os.makedirs(DATABASE_FOLDER)

def save_uploaded_file(uploaded_file):
    file_path = os.path.join(DATABASE_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def queryDB(query_str, db_path):
    engine = create_engine(f"sqlite:///{db_path}")
    sql_database = SQLDatabase(engine, include_tables=["product_master", "inventory"])

    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=["product_master", "inventory"],
        verbose=True
    )

    response = query_engine.query(query_str)
    return response

# Streamlit UI
st.title("Query the database - TEXT to SQL")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask me a question !"}]

uploaded_file = st.file_uploader("Upload a .db file", type="db")

if uploaded_file:
    db_path = save_uploaded_file(uploaded_file)
    st.session_state.db_path = db_path

if st.session_state.get("db_path"):
    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = queryDB(prompt, st.session_state.db_path)
                st.write(response.response)
                pprint_response(response, show_source=True)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)
else:
    st.write("Please upload a .db file to proceed.")
