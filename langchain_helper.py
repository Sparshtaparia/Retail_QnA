import os
import re
from dotenv import load_dotenv
from urllib.parse import quote_plus

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_experimental.sql import SQLDatabaseChain
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector

from few_shots import few_shots

# Load environment
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")


# Clean SQL
def clean_sql_query(query: str) -> str:
    """
    Remove unwanted prefixes, code fences, invisible quotes, and whitespace.
    """
    query = re.sub(r"```(?:sql)?", "", query, flags=re.IGNORECASE)  # remove code fences
    query = re.sub(r"^\s*sql\s*", "", query, flags=re.IGNORECASE)   # remove leading 'sql'
    query = query.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    return query.strip()


# Extract SQL safely from LangChain intermediate_steps
def extract_sql_from_steps(intermediate_steps) -> str:
    """
    Recursively extract the first string that looks like SQL (contains SELECT)
    from LangChain intermediate_steps.
    """
    if not intermediate_steps:
        raise ValueError("No intermediate steps returned from chain")

    step = intermediate_steps[0]

    def find_sql(obj):
        if isinstance(obj, str):
            if "SELECT" in obj.upper():  # crude check for SQL
                return obj
            return ""
        elif isinstance(obj, dict):
            for key, value in obj.items():
                result = find_sql(value)
                if result:
                    return result
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                result = find_sql(item)
                if result:
                    return result
        return ""

    sql = find_sql(step)
    if not sql:
        raise ValueError("Could not find SQL in intermediate_steps")
    return sql

# Build the chain
def get_few_shot_db_chain():
    db_user = "root"
    db_password = os.getenv("DB_PASSWORD")
    db_host = "localhost"
    db_name = "atliq_tshirts"

    encoded_password = quote_plus(str(db_password))
    db = SQLDatabase.from_uri(
        f"mysql+pymysql://{db_user}:{encoded_password}@{db_host}:3306/{db_name}",
        sample_rows_in_table_info=3
    )

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1024
    )

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    to_vectorize = [" ".join(str(v) for v in example.values()) for example in few_shots]
    vectorstore = Chroma.from_texts(texts=to_vectorize, embedding=embeddings, metadatas=few_shots)

    example_selector = SemanticSimilarityExampleSelector(vectorstore=vectorstore, k=2)

    mysql_prompt = """
You are a MySQL expert.

Given a question, generate a syntactically correct MySQL query.

Rules:
- Use only columns that exist
- Do not hallucinate columns
- Use SUM(stock_quantity) when counting inventory
- Use SUM(price * stock_quantity) when calculating inventory value
- Use LEFT JOIN for discounts
- DO NOT wrap SQL in markdown or ```sql blocks
- Return ONLY the SQL query
"""

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
        template="""Question: {Question}
SQLQuery: {SQLQuery}
SQLResult: {SQLResult}
Answer: {Answer}"""
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=mysql_prompt,
        suffix="Question: {input}\nSQLQuery:",
        input_variables=["input"]
    )

    chain = SQLDatabaseChain.from_llm(
        llm=llm,
        db=db,
        verbose=True,
        prompt=few_shot_prompt,
        use_query_checker=False,  # we will clean SQL manually
        return_intermediate_steps=True
    )

    return chain


# Run text-to-SQL safely
def run_text_to_sql(chain, db, question):
    """
    Generate SQL from question using chain, clean it, execute on DB,
    and return numeric value.
    """
    response = chain.invoke({"query": question})

    # Extract the actual SQL string
    raw_sql = extract_sql_from_steps(response.get("intermediate_steps", []))
    cleaned_sql = clean_sql_query(raw_sql)

    # Execute on DB
    result = db.run(cleaned_sql)

    # Extract numeric value
    numeric_answer = result[0][0] if result and len(result) > 0 else None
    return numeric_answer