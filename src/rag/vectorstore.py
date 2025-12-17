"""
Vector Store Manager for SHL Assessments
Handles creation and loading of ChromaDB/FAISS vector stores
"""

from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStoreManager:
    def __init__(self, embeddings, persist_directory="data/processed/vectorstore"):
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.vectorstore = None
    
    def create_vectorstore(self, documents: List[Document], use_chroma=True):
        """
        Create a new vector store from documents.
        
        Args:
            documents: List of LangChain Document objects
            use_chroma: If True, use ChromaDB (persistent), else use FAISS
        
        Returns:
            Vector store object
        """
        logger.info(f"Creating vector store with {len(documents)} documents...")
        
        if use_chroma:
            # ChromaDB - Persistent storage
            logger.info("Using ChromaDB (persistent storage)")
            
            # Remove existing directory if it exists
            if os.path.exists(self.persist_directory):
                import shutil
                shutil.rmtree(self.persist_directory)
                logger.info("Removed existing vectorstore")
            
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name="shl_assessments"
            )
            logger.info(f"✅ ChromaDB created at: {self.persist_directory}")
            
        else:
            # FAISS - In-memory, faster but non-persistent
            logger.info("Using FAISS (in-memory storage)")
            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            # Save FAISS index
            os.makedirs(self.persist_directory, exist_ok=True)
            faiss_path = os.path.join(self.persist_directory, "faiss_index")
            self.vectorstore.save_local(faiss_path)
            logger.info(f"✅ FAISS index saved at: {faiss_path}")
        
        return self.vectorstore
    
    def load_vectorstore(self, use_chroma=True):
        """Load existing vector store"""
        logger.info("Loading existing vector store...")
        
        if use_chroma:
            if not os.path.exists(self.persist_directory):
                raise ValueError(f"Vectorstore not found at: {self.persist_directory}")
            
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="shl_assessments"
            )
            logger.info("✅ ChromaDB loaded")
            
        else:
            faiss_path = os.path.join(self.persist_directory, "faiss_index")
            if not os.path.exists(faiss_path):
                raise ValueError(f"FAISS index not found at: {faiss_path}")
            
            self.vectorstore = FAISS.load_local(
                faiss_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("✅ FAISS index loaded")
        
        return self.vectorstore
    
    def similarity_search(self, query: str, k: int = 10):
        """Perform similarity search"""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Call create_vectorstore() or load_vectorstore() first.")
        
        return self.vectorstore.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 10):
        """Perform similarity search with scores"""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized.")
        
        return self.vectorstore.similarity_search_with_score(query, k=k)


def main():
    """Build and save vector store"""
    import pandas as pd
    from embeddings import AssessmentEmbedder
    
    # Load catalog
    df = pd.read_csv('data/raw/shl_assessments.csv')
    logger.info(f"Loaded {len(df)} assessments")
    
    # Create embeddings
    embedder = AssessmentEmbedder()
    documents, _ = embedder.embed_catalog(df)
    
    # Create vector store
    vs_manager = VectorStoreManager(
        embeddings=embedder.embeddings,
        persist_directory="data/processed/vectorstore"
    )
    vectorstore = vs_manager.create_vectorstore(documents, use_chroma=True)
    
    # Test retrieval
    test_query = "Java developer with collaboration skills"
    results = vs_manager.similarity_search(test_query, k=5)
    
    print("\n" + "="*80)
    print(f"TEST QUERY: {test_query}")
    print("="*80)
    print(f"\nTop 5 results:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.metadata['name']}")
        print(f"   Type: {doc.metadata['test_type']} | Category: {doc.metadata['category']}")
        print(f"   URL: {doc.metadata['url']}")
    print("="*80)
    
    logger.info("✅ Vector store built and tested successfully!")


if __name__ == "__main__":
    main()
