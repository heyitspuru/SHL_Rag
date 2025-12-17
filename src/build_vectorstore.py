"""
Build the vector store for SHL assessments
"""

import sys
import os
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag.embeddings import AssessmentEmbedder
from rag.vectorstore import VectorStoreManager

def build_vectorstore():
    """Build and save the vector store"""
    print("="*80)
    print("BUILDING VECTOR STORE")
    print("="*80)
    
    # Load catalog
    catalog_path = 'data/raw/shl_assessments.csv'
    print(f"\n1. Loading catalog from {catalog_path}...")
    df = pd.read_csv(catalog_path)
    print(f"   ✅ Loaded {len(df)} assessments")
    
    # Create embeddings
    print("\n2. Creating embeddings...")
    embedder = AssessmentEmbedder(model_name="all-MiniLM-L6-v2")
    documents, _ = embedder.embed_catalog(df)
    print(f"   ✅ Created embeddings for {len(documents)} documents")
    
    # Build vector store
    print("\n3. Building ChromaDB vector store...")
    vs_manager = VectorStoreManager(
        embeddings=embedder.embeddings,
        persist_directory="data/processed/vectorstore"
    )
    vectorstore = vs_manager.create_vectorstore(documents, use_chroma=True)
    print(f"   ✅ Vector store created and persisted")
    
    # Test retrieval
    print("\n4. Testing retrieval...")
    test_query = "Java programming skills"
    results = vs_manager.similarity_search(test_query, k=3)
    print(f"   Test query: '{test_query}'")
    print(f"   Top 3 results:")
    for i, doc in enumerate(results, 1):
        print(f"   {i}. {doc.metadata['name']} ({doc.metadata['test_type']})")
    
    print("\n" + "="*80)
    print("✅ VECTOR STORE BUILD COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    build_vectorstore()
