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
client = Groq(api_key="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")

# Load the dataset
dataset_path = 'https://raw.githubusercontent.com/noumantechie/RagApplication/main/lungcaner/dataseter.csv'  # Ensure this file is uploaded
df = pd.read_csv(dataset_path)

# Prepare embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # Open-source embedding model

# Convert dataset rows to embeddings
def row_to_text(row):
    return " ".join(f"{col}: {val}" for col, val in row.items())

df['text'] = df.apply(row_to_text, axis=1)
embeddings = np.vstack(df['text'].apply(lambda x: model.encode(x)).to_numpy())

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
    return chat_completion.choices[0].message.content

# Streamlit interface enhancements

# Set custom page configuration
st.set_page_config(page_title="Medical Query Answering System", layout="centered")

# Add a background color and some padding for the title
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
        }
        .stTitle {
            font-size: 32px;
            font-weight: bold;
            color: #2a3d66;
            padding-bottom: 20px;
        }
        .stTextInput input {
            font-size: 16px;
            padding: 10px;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .stWrite {
            font-size: 18px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Title of the app
st.markdown("<h1 class='stTitle'>Medical Query Answering System</h1>", unsafe_allow_html=True)

# Explanation text
st.write("Enter a query below and get a detailed response based on the dataset.")

# User input query
query = st.text_input("Your Query", "")

# Add a button to trigger query execution
if st.button("Get Answer"):
    if query:
        response = rag_pipeline(query)
        st.markdown(f"### Response: \n{response}")
    else:
        st.warning("Please enter a query to get a response.")
