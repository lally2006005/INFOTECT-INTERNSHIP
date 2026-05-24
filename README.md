# INFOTECT-INTERNSHIP W1
# Infotect internship week 1

# Prerequisites
# First, ensure you have the required packages installed

pip install langchain langchain-openai langchain-community pinecone-client unstructured pydantic

# You will also need to set up your environment variables

export OPENAI_API_KEY="your-openai-key"
export PINECONE_API_KEY="your-pinecone-key"

# Week 1: Ingestion Pipeline Implementation

import os
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec


# 1. INITIALIZATION & SETUP

# Ensure API keys are present
if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("PINECONE_API_KEY"):
    raise ValueError("Please set BOTH OPENAI_API_KEY and PINECONE_API_KEY environment variables.")

# Initialize Pinecone Client
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

INDEX_NAME = "documind-enterprise-index"
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dimensions
DIMENSION = 1536

# Create the Pinecone Index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating Pinecone index: {INDEX_NAME}...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # Adjust region based on your preference
        )
    )
else:
    print(f"Index '{INDEX_NAME}' already exists.")

# Connect to the target index
index = pc.Index(INDEX_NAME)


# 2. DOCUMENT LOADING (Unstructured.io)

# Replace 'your_policy_document.pdf' with your actual file path
file_path = "your_policy_document.pdf" 

print(f"Loading document using Unstructured: {file_path}...")
loader = UnstructuredFileLoader(file_path, mode="elements")
documents = loader.load()
print(f"Loaded {len(documents)} elements from the document.")


# 3. SOPHISTICATED CHUNKING STRATEGY

print("Splitting document text into sophisticated chunks...")
# Setting a clean chunk size and overlap for context retention
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} text chunks.")


# 4. EMBEDDING GENERATION & PINECONE UPSERT

print("Initializing OpenAI Embeddings...")
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

print("Vectorizing and Upserting chunks to Pinecone...")
vectors_to_upsert = []

for idx, chunk in enumerate(chunks):
    # Extract structural metadata for later citation use (Week 4 requirement)
    page_number = chunk.metadata.get("page_number", 1)
    source_doc = chunk.metadata.get("source", file_path)
    
    # Generate the vector embedding for the text chunk
    vector = embeddings.embed_query(chunk.page_content)
    
    # Structure the payload payload for Pinecone
    vectors_to_upsert.append({
        "id": f"chunk_id_{idx}",
        "values": vector,
        "metadata": {
            "text": chunk.page_content,
            "page_number": page_number,
            "source": source_doc
        }
    })
    
    # Upsert in batches of 100 to prevent hitting payload limit size limits
    if len(vectors_to_upsert) == 100 or idx == len(chunks) - 1:
        index.upsert(vectors=vectors_to_upsert)
        vectors_to_upsert = []

# print("Ingestion pipeline successfully completed!")

# Verification and Testing Focus
# Per your requirement to verify if a query for a specific policy (e.g., "How do I get money back?") successfully retrieves the relevant chunks (e.g., "Refund Policy"), use this verification script


# WEEK 1 VERIFICATION SCRIPT

def verify_retrieval(query_text: str):
    print(f"\n[Verification] Running Query: '{query_text}'")
    
    # 1. Embed the testing query
    query_vector = embeddings.embed_query(query_text)
    
    # 2. Query Pinecone for top 3 matching chunks
    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )
    
    # 3. Verify results
    print("\n--- Top Retrieved Chunks ---")
    for match in results["matches"]:
        score = match["score"]
        text = match["metadata"]["text"]
        page = match["metadata"]["page_number"]
        print(f"\n[Score: {score:.4f}] [Page: {page}]")
        print(f"Content: {text[:200]}...")  # Show first 200 characters

# Execute verification test
verify_retrieval("How do I get money back?")
