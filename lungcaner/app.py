# Import libraries
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from groq import Groq
import streamlit as st
import csv

# Set up Groq API
try:
    secrets = st.secrets["groq_api_key"]
except KeyError:
    st.error("API Key not found in secrets. Please check the Streamlit Secrets configuration.")

# Set up Groq API with the secret key
client = Groq(api_key=secrets)

# Load the dataset
dataset_path = 'https://raw.githubusercontent.com/noumantechie/RagApplication/main/lungcaner/dataseter.csv'  # Ensure this file is uploaded
try:
    df = pd.read_csv(dataset_path)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Check the columns of the dataframe to ensure the correct ones exist
st.write("Dataset Columns:", df.columns)

# Prepare embeddings (caching embeddings to avoid recomputation)
model = SentenceTransformer('all-MiniLM-L6-v2')  # Open-source embedding model

@st.cache_data
def compute_embeddings(df):
    # Convert dataset rows to embeddings
    def row_to_text(row):
        return " ".join(f"{col}: {val}" for col, val in row.items())

    df['text'] = df.apply(row_to_text, axis=1)
    embeddings = np.vstack(df['text'].apply(lambda x: model.encode(x)).to_numpy())
    return embeddings

embeddings = compute_embeddings(df)

# Define retrieval function
def retrieve_relevant_rows(query, top_n=3):
    query_embedding = model.encode(query)
    similarities = cosine_similarity([query_embedding], embeddings).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    return df.iloc[top_indices]

# RAG Functionality
def rag_pipeline(query):
    # Step 1: Retrieve relevant rows
    retrieved_rows = retrieve_relevant_rows(query, top_n=3)

    # Step 2: Combine retrieved data for the Groq model
    retrieved_text = " ".join(retrieved_rows['text'].tolist())
    input_to_groq = f"Context: {retrieved_text} \nQuestion: {query}"

    # Step 3: Use Groq for text generation
    try:
        chat_completion = client.chat.completions.create(
            messages=[{
                "role": "system",
                "content": "You are an expert in analyzing medical data related to lung cancer.",
            },
            {
                "role": "user",
                "content": input_to_groq,
            }],
            model="llama3-8b-8192",  # Use Groq's Llama model
        )
        response_content = chat_completion.choices[0].message.content
        if response_content:
            return response_content
        else:
            return "No valid response from the model."
    except Exception as e:
        st.error(f"Error in Groq API call: {e}")
        return "There was an issue processing your request."

# Streamlit interface
st.title("Medical Query Answering System")
st.write("Enter a query below and get a detailed response based on the dataset.")

# User input query
query = st.text_input("Your Query", "")

# Handle user input and show results
if query:
    if len(query.strip()) > 3:
        with st.spinner('Generating response...'):
            response = rag_pipeline(query)
        st.write("Response:", response)
    else:
        st.warning("Please enter a longer query.")
