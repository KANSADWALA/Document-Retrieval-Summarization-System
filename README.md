# 🗃️Document Retrieval & Summarization System

## Overview

This system provides an efficient and user-friendly way to search for academic papers from the Arxiv repository and generate summaries for the retrieved documents. This tool is especially useful for researchers, students, and academics looking to quickly grasp the essence of papers in various fields. The system allows users to enter a search query, retrieves relevant papers, and automatically generates concise summaries of each paper to save time and effort.

## Features

<ul>
  <li><strong>Document Retrieval:</strong>  Efficiently search and retrieve academic papers from the arXiv dataset based on user queries.</li>
  
  <li><strong>Summarization:</strong> Generate concise summaries of paper abstracts</li>
  
  <li><strong>User-friendly Interface:</strong> Interactive web interface for searching documents and viewing summaries.</li>
</ul>

## Technologies Used

<ul>
  <li><strong>sentence_transformers:</strong> Provides pre-trained models to generate sentence embeddings, which are used for converting text documents into vector representations.</li>

  <li><strong>faiss:</strong> A library developed by Facebook AI Research for efficient similarity search and clustering of dense vectors, used for fast document retrieval.</li>
  
  <li><strong>transformers:</strong> A library by Hugging Face offering a wide range of pre-trained models for various NLP tasks. Used for BART, which is utilized for text summarization.</li>

  <li><strong>torch:</strong> A deep learning framework used for training and deploying models, including the SentenceTransformer and BART models.</li>

  <li><strong>langchain:</strong> Facilitates the creation of workflows and chains for integrating various NLP tasks, enhancing modularity and reusability of code.</li>

  <li><strong>spacy:</strong> A robust library for natural language processing, used here for named entity recognition (NER) and text preprocessing.</li>
  
  <li><strong>gensim:</strong> A library for topic modeling and evaluating coherence scores, used here for advanced topic modeling and assessment of topic coherence.</li>
  
  <li><strong>sklearn:</strong> Includes tools for machine learning and text processing, such as TF-IDF vectorization, PCA for dimensionality reduction, and LDA for topic modeling.</li>

  <li><strong>streamlit:</strong> An open-source app framework that allows for the creation of interactive web applications, used to build the user interface for querying and displaying search results and summaries.</li>
</ul>

## Project Structure

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<p>The directory structure of the project is as follows:</p>

<pre><code class="bash">

  Document Retrieval & Summarization System/
  │
  ├── kaggle.json               # Kaggle API key for dataset download
  │
  │──Document Retrieval & Summarization System.ipynb
  │
  ├── app.py                    # Streamlit web application
  │
  umentation
  ├── requirements.txt          # Python package dependencies
  │
  ├── README.md                 # Project overview and doc
  
</code></pre>

</body>
</html> 

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.


## Contact
For any questions or suggestions, please open an issue or contact me at <a href="mailto:shubhamkansadwala@gmail.com">shubhamkansadwala@gmail.com</a>
.
<hr></hr>
