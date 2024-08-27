import streamlit as st
import json
import pandas as pd
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import BartTokenizer, BartForConditionalGeneration

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model = SentenceTransformer('all-mpnet-base-v2', device=device)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)

# Load dataset
def load_data():
    with open('arxiv-metadata-oai-snapshot.json', 'r') as file:
        arxiv_data = [json.loads(line) for i, line in enumerate(file) if i < 1000]
    df = pd.DataFrame(arxiv_data)[['id', 'authors', 'title', 'categories', 'abstract']]
    df['content'] = df['title'] + ' ' + df['abstract']
    return df

df = load_data()

# Create FAISS index
def create_faiss_index(df):
    documents = df['content'].tolist()
    embeddings = embedding_model.encode(documents, convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # Use IndexFlatIP for simplicity
    index.add(embeddings)
    return index

index = create_faiss_index(df)

# Search function
def search_documents(query, model, index, df, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

    distances, indices = index.search(query_embedding, top_k)
    valid_indices = indices[0]  # Use all indices returned by search
    results = df.iloc[valid_indices].copy()
    results['similarity'] = distances[0] * 100  # Convert to percentage
    results = results.sort_values('similarity', ascending=False)  # Sort in descending order
    return results

# Summary generation
def generate_summary(text):
    inputs = tokenizer([text], max_length=1024, return_tensors="pt").to(device)
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=100, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Streamlit app layout

st.set_page_config(page_title="Document Retrieval & Summarization System", page_icon=":books:")
st.title("Arxiv Paper Search and Summarization")

query = st.text_input("Enter your search query:")
if st.button("Search"):
    st.write("### Search Query Result:")
    st.write(f"**Search Query:** {query}")

    results = search_documents(query, embedding_model, index, df, top_k=1)

    if not results.empty:
        search_result = results.iloc[0]
        st.write(f"**Title:** {search_result['title']}")
        st.write(f"**Authors:** {search_result['authors']}")
        st.write(f"**Abstract:** {search_result['abstract']}")
        summary = generate_summary(search_result['abstract'])
        st.write(f"**Summary:** {summary}")

        # Exclude the search result from similar documents
        similar_results = search_documents(query, embedding_model, index, df, top_k=5)
        similar_results = similar_results[similar_results['id'] != search_result['id']]

        st.write("### Similar Documents:")
        if not similar_results.empty:
            for i, (index, row) in enumerate(similar_results.iterrows(), start=1):
                st.write(f"{i}. **{row['title']}**")
                st.write(f"   **Authors:** {row['authors']}")
                st.write(f"   **Abstract:** {row['abstract']}")
                summary = generate_summary(row['abstract'])
                st.write(f"   **Summary:** {summary}")
                st.write(f"   **Similarity:** {row['similarity']:.2f}%")
        else:
            st.write("No similar documents found.")

    else:
        st.write("No exact match found for the search query.")

st.write("### Dataset Overview:")
st.dataframe(df.head(10))
