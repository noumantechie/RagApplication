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


# Set page config
st.set_page_config(page_title="Medical Query Answering System", layout="centered")

# Add custom CSS styling for professional look
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f0f0;
            font-family: 'Roboto', sans-serif;
        }
        .stTitle {
            color: #2a3d66;
            font-size: 40px;
            font-weight: 600;
            margin-bottom: 20px;
        }
        .stTextInput input {
            font-size: 16px;
            padding: 15px;
            width: 100%;
            border-radius: 8px;
            border: 1px solid #ccc;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .stButton button {
            background-color: #007BFF;
            color: white;
            font-size: 18px;
            padding: 15px 30px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .stButton button:hover {
            background-color: #0056b3;
        }
        .stButton button:active {
            background-color: #003f7f;
        }
        .stWrite {
            font-size: 18px;
            color: #333;
            line-height: 1.6;
        }
        .stCard {
            background-color: #ffffff;
            padding: 30px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            border-radius: 12px;
        }
        .response-container {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Title of the app with a stylish header
st.markdown("<h1 class='stTitle'>Medical Query Answering System</h1>", unsafe_allow_html=True)

# Add some explanation text
st.markdown("<p style='font-size:18px; color:#555;'>Enter a medical query below, and get a detailed response based on the dataset.</p>", unsafe_allow_html=True)

# Main container for user input and button
with st.container():
    st.markdown("<div class='stCard'>", unsafe_allow_html=True)
    # User input query with a cleaner input field
    query = st.text_input("Your Query", placeholder="Type your query here...")
    
    # Add a styled button
    if st.button("Get Answer"):
        if query:
            response = rag_pipeline(query)
            st.markdown(f"<div class='response-container'><h4>Response:</h4><p>{response}</p></div>", unsafe_allow_html=True)
        else:
            st.warning("Please enter a query to get a response.")

    st.markdown("</div>", unsafe_allow_html=True)
