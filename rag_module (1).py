# rag_module.py
import os
import json
import requests
import faiss
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from time import sleep

# ─── Configuration ──────────────────────────
PDF_FOLDER = "/content/drive/MyDrive/Colab Notebooks/PathoBuddy/data/pdfs"
INDEX_DIR = "/content/drive/MyDrive/Colab Notebooks/PathoBuddy/embeddings/faiss_index"
API_KEY = os.getenv("GOOGLE_API_KEY")

# ✅ CORRECTED Gemini embeddings endpoint
EMBED_URL = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={API_KEY}"

# ─── Helper: Get Embedding ─────────────────
def get_embedding(text: str) -> list[float]:
    """Get embedding from Gemini API with correct format"""
    # Truncate to avoid exceeding request size limits
    if len(text) > 2048:  # Gemini embedding limit is ~2048 tokens
        text = text[:2048]
    
    # ✅ CORRECT payload format for Gemini
    payload = {
        "model": "models/embedding-001",
        "content": {
            "parts": [{"text": text}]
        }
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        resp = requests.post(EMBED_URL, json=payload, headers=headers, timeout=30)
        
        if resp.status_code != 200:
            print(f"❌ API Response: {resp.status_code} - {resp.text}")
            raise RuntimeError(f"Embedding API Error {resp.status_code}: {resp.text}")
        
        data = resp.json()
        
        # ✅ CORRECT response parsing for Gemini
        if "embedding" in data and "values" in data["embedding"]:
            return data["embedding"]["values"]
        else:
            print(f"❌ Unexpected response format: {data}")
            raise RuntimeError(f"Unexpected response format: {data}")
            
    except requests.exceptions.RequestException as e:
        # Simple retry on failure
        print(f"⚠️ Request failed: {e}. Retrying after 5 seconds...")
        sleep(5)
        resp = requests.post(EMBED_URL, json=payload, headers=headers, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Embedding API Error after retry: {resp.status_code}: {resp.text}")
        data = resp.json()
        if "embedding" in data and "values" in data["embedding"]:
            return data["embedding"]["values"]
        else:
            raise RuntimeError(f"Unexpected response format after retry: {data}")

# ─── 1) Build the FAISS Index ─────────────────
def build_index(pdf_folder=PDF_FOLDER, index_dir=INDEX_DIR):
    """Build FAISS index from PDFs"""
    print("📂 Loading PDF documents...")
    docs = []
    
    # Check if PDF folder exists
    if not os.path.exists(pdf_folder):
        raise RuntimeError(f"PDF folder not found: {pdf_folder}")
    
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise RuntimeError(f"No PDF files found in: {pdf_folder}")
    
    print(f"📄 Found {len(pdf_files)} PDF files")
    
    for fn in pdf_files:
        try:
            loader = PyPDFLoader(os.path.join(pdf_folder, fn))
            docs.extend(loader.load())
            print(f"✅ Loaded: {fn}")
        except Exception as e:
            print(f"❌ Failed to load {fn}: {e}")

    if not docs:
        raise RuntimeError("No documents loaded successfully")

    print("✂️ Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    
    print(f"📝 Created {len(chunks)} chunks")

    embeddings, texts = [], []
    print(f"🔄 Processing {len(chunks)} chunks for embeddings...")
    
    for idx, doc in enumerate(chunks, 1):
        txt = doc.page_content.strip()
        # Clean text to remove artifacts
        txt = '\n'.join(line for line in txt.splitlines() if len(line.strip()) > 5)  # Remove short/noisy lines
        txt = txt.replace("~~", "").replace("\f", "").replace("~~~~~", "")  # Remove common PDF artifacts
        if not txt or len(txt) < 10:  # Skip very short chunks
            continue
            
        try:
            print(f"Processing chunk {idx}/{len(chunks)}", end="\r")
            emb = get_embedding(txt)
            embeddings.append(emb)
            texts.append(txt)
            
        except Exception as e:
            print(f"\n❌ Embedding failed for chunk {idx}: {e}")
            continue

    if not embeddings:
        raise RuntimeError("❌ No embeddings generated; check API key and documents.")

    print(f"\n✅ Generated {len(embeddings)} embeddings")
    
    # Build FAISS index
    print("🏗️ Building FAISS index...")
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    matrix = np.array(embeddings, dtype="float32")
    index.add(matrix)

    # Save index and texts
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    
    with open(os.path.join(index_dir, "texts.json"), "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    
    print(f"💾 FAISS index saved at: {index_dir}")
    print(f"📊 Index contains {len(texts)} chunks with {dim}-dimensional embeddings")

# ─── 2) Load the Index ─────────────────
def load_index(index_dir=INDEX_DIR):
    """Load existing FAISS index"""
    index_path = os.path.join(index_dir, "index.faiss")
    texts_path = os.path.join(index_dir, "texts.json")
    
    if not os.path.exists(index_path) or not os.path.exists(texts_path):
        raise RuntimeError(f"Index not found at {index_dir}. Run build_index() first.")
    
    idx = faiss.read_index(index_path)
    with open(texts_path, encoding="utf-8") as f:
        texts = json.load(f)
    
    return idx, texts

# ─── 3) Retrieve Relevant Snippets ─────────────────
def retrieve_relevant(question: str, k=5, index_dir=INDEX_DIR) -> list[str]:  # Increased k for more context
    """Retrieve top-k relevant text chunks for a question"""
    print(f"🔍 Searching for: {question}")
    
    # Get query embedding
    qvec = np.array([get_embedding(question)], dtype="float32")
    
    # Load index and search
    idx, texts = load_index(index_dir)
    dists, ids = idx.search(qvec, k)
    
    results = []
    for i, dist in zip(ids[0], dists[0]):
        if i != -1:  # Valid result
            results.append(texts[i])
    
    print(f"✅ Found {len(results)} relevant chunks")
    return results

# ─── Test API Connection ─────────────────
def test_api():
    """Test if Gemini API is working"""
    if not API_KEY:
        raise ValueError("❌ GOOGLE_API_KEY environment variable not set!")
    
    print("🧪 Testing Gemini API connection...")
    try:
        test_emb = get_embedding("This is a test sentence for pathology.")
        print(f"✅ API working! Embedding dimension: {len(test_emb)}")
        return True
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

# ─── Quick Test ─────────────────
if __name__ == "__main__":
    # Test API first
    if not test_api():
        print("❌ Fix API issues before building index")
        exit(1)
    
    print("\n🏗️ Building FAISS index...")
    try:
        build_index()
        print("\n🔍 Testing retrieval...")
        results = retrieve_relevant("Which stain is used for tuberculosis diagnosis?", k=5)
        
        print("\n📋 Retrieved snippets:")
        for i, snippet in enumerate(results, 1):
            print(f"\n--- Snippet {i} ---")
            print(snippet[:200] + "..." if len(snippet) > 200 else snippet)
            
    except Exception as e:
        print(f"❌ Error: {e}")