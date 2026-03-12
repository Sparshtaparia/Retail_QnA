import streamlit as st
from langchain_helper import get_few_shot_db_chain, run_text_to_sql
from langchain_community.utilities import SQLDatabase
from urllib.parse import quote_plus
import os

st.title("AtliQ T Shirts: Database Q&A 👕")

question = st.text_input("Question:")

if question:
    chain = get_few_shot_db_chain()

    # DB object
    db_user = "root"
    db_password = os.getenv("DB_PASSWORD")
    db_host = "localhost"
    db_name = "atliq_tshirts"
    encoded_password = quote_plus(str(db_password))
    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{encoded_password}@{db_host}:3306/{db_name}")

    try:
        numeric_answer = run_text_to_sql(chain, db, question)
        st.subheader("Answer (numeric value)")
        st.write(numeric_answer)

    except Exception as e:
        st.error(f"❌ Error: {e}")