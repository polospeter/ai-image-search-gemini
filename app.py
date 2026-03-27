import os
import subprocess

import streamlit as st
from google import genai
from PIL import Image
import chromadb
from chromadb.config import Settings

DB_PATH = "./chroma_db"
COLLECTION_NAME = "images"
EMBED_MODEL = "gemini-embedding-2-preview"


@st.cache_resource
def get_clients():
    genai_client = genai.Client(vertexai=True, project="ai-search-agent-491514", location="us-central1")
    chroma_client = chromadb.PersistentClient(
        path=DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = chroma_client.get_or_create_collection(
        COLLECTION_NAME,
        embedding_function=None,
    )
    return genai_client, collection


st.title("Image Search")
st.caption("Search your image library using natural language.")

genai_client, collection = get_clients()

if collection.count() == 0:
    st.warning("No images indexed yet. Run `python3 index_images.py` first.")
    st.stop()

with st.sidebar:
    top_n = st.slider("Results to show", min_value=3, max_value=30, value=12)
    st.caption(f"{collection.count()} images indexed")

query = st.text_input("Describe what you're looking for", placeholder="e.g. steampunk robot in snow")
search = st.button("Search", type="primary")

if search and query.strip():
    with st.spinner("Searching..."):
        response = genai_client.models.embed_content(model=EMBED_MODEL, contents=[query])
        query_vector = list(response.embeddings[0].values)

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_n,
            include=["metadatas", "distances"],
        )

    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    cols = st.columns(3)
    for idx, (meta, dist) in enumerate(zip(metadatas, distances)):
        path = meta["path"]
        filename = meta["filename"]
        col = cols[idx % 3]

        with col:
            try:
                img = Image.open(path)
                img.thumbnail((400, 400))
                st.image(img, use_container_width=True)
            except Exception:
                st.warning("Image unavailable")

            st.caption(filename)
            similarity = max(0, 100 - dist * 10)
            st.caption(f"Similarity: {similarity:.0f}%")

            if os.path.exists(path):
                if st.button("Open in Finder", key=path):
                    subprocess.Popen(["open", "-R", path])
            else:
                st.caption("File not found")
