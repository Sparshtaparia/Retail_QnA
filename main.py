import streamlit as st
from langchain_helper import get_few_shot_db_chain, clean_sql_query, run_text_to_sql
from langchain_community.utilities import SQLDatabase
from urllib.parse import quote_plus
import os

st.title("LLM-Powered SQL Database Assistant ")

question = st.text_input("Type your inventory or sales question here: ")

if question:
    # It is better to initialize the chain once outside the 'if' 
    # or use st.cache_resource to prevent reloading on every click
    chain = get_few_shot_db_chain()

    try:
        # We only want the numeric answer
        answer = run_text_to_sql(chain, question)
        
        # Display ONLY the answer
        st.subheader("Answer")
        st.header(f"{answer}") 

    except Exception as e:
        st.error(f"Error: {e}")