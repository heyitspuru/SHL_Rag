"""
Assessment Retriever with advanced retrieval strategies
"""

from langchain.schema import Document
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AssessmentRetriever:
    def __init__(self, vectorstore, llm=None):
        self.vectorstore = vectorstore
        self.llm = llm
    
    def retrieve_basic(self, query: str, k: int = 20) -> List[Document]:
        """
        Basic semantic similarity retrieval.
        Returns top-k most similar assessments.
        """
        logger.info(f"Retrieving top {k} assessments for query: {query[:100]}...")
        docs = self.vectorstore.similarity_search(query, k=k)
        return docs
    
    def retrieve_with_score(self, query: str, k: int = 20) -> List[Tuple[Document, float]]:
        """Retrieve with similarity scores"""
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        return docs_with_scores
    
    def retrieve_mmr(self, query: str, k: int = 20, fetch_k: int = 50, lambda_mult: float = 0.7) -> List[Document]:
        """
        Maximum Marginal Relevance (MMR) retrieval.
        Balances relevance and diversity.
        
        Args:
            query: Search query
            k: Number of documents to return
            fetch_k: Number of documents to fetch initially
            lambda_mult: 0-1, where 1 is max relevance, 0 is max diversity
        
        Returns:
            Diverse set of relevant documents
        """
        logger.info(f"MMR retrieval (k={k}, fetch_k={fetch_k}, λ={lambda_mult})")
        
        try:
            docs = self.vectorstore.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult
            )
            return docs
        except Exception as e:
            logger.warning(f"MMR failed: {e}, falling back to basic retrieval")
            return self.retrieve_basic(query, k=k)
    
    def enhance_query(self, query: str) -> str:
        """
        Use LLM to enhance/expand the query for better retrieval.
        Extracts key skills, requirements, and context.
        """
        if not self.llm:
            return query
        
        try:
            from langchain.prompts import PromptTemplate
            
            prompt = PromptTemplate(
                input_variables=["query"],
                template="""Analyze this job query and extract key assessment requirements:
                
Query: {query}

Extract:
1. Technical skills needed
2. Soft skills/behavioral traits
3. Cognitive abilities
4. Domain/industry context

Provide a concise, keyword-rich summary for assessment matching:
"""
            )
            
            enhanced = self.llm.invoke(prompt.format(query=query))
            if hasattr(enhanced, 'content'):
                enhanced = enhanced.content
            
            logger.info("Query enhanced with LLM")
            return f"{query}\n\n{enhanced}"
            
        except Exception as e:
            logger.warning(f"Query enhancement failed: {e}")
            return query
    
    def filter_by_type(self, docs: List[Document], test_types: List[str]) -> List[Document]:
        """Filter documents by test type"""
        return [doc for doc in docs if doc.metadata.get('test_type') in test_types]
    
    def get_type_distribution(self, docs: List[Document]) -> dict:
        """Get distribution of test types in results"""
        distribution = {}
        for doc in docs:
            test_type = doc.metadata.get('test_type', 'Unknown')
            distribution[test_type] = distribution.get(test_type, 0) + 1
        return distribution


def main():
    """Test retriever"""
    import pandas as pd
    from embeddings import AssessmentEmbedder
    from vectorstore import VectorStoreManager
    
    # Load embeddings and vector store
    embedder = AssessmentEmbedder()
    vs_manager = VectorStoreManager(embedder.embeddings)
    
    try:
        vectorstore = vs_manager.load_vectorstore(use_chroma=True)
    except:
        logger.info("Vector store not found, creating new one...")
        df = pd.read_csv('data/raw/shl_assessments.csv')
        documents, _ = embedder.embed_catalog(df)
        vectorstore = vs_manager.create_vectorstore(documents, use_chroma=True)
    
    # Create retriever
    retriever = AssessmentRetriever(vectorstore)
    
    # Test queries
    test_queries = [
        "Java developer with collaboration skills",
        "Sales representative with customer service experience",
        "Entry-level data analyst position"
    ]
    
    for query in test_queries:
        print("\n" + "="*80)
        print(f"QUERY: {query}")
        print("="*80)
        
        # Basic retrieval
        docs = retriever.retrieve_basic(query, k=5)
        print(f"\n📊 Top 5 Basic Retrieval Results:")
        for i, doc in enumerate(docs, 1):
            print(f"{i}. {doc.metadata['name']} ({doc.metadata['test_type']})")
        
        # MMR retrieval
        docs_mmr = retriever.retrieve_mmr(query, k=5, lambda_mult=0.7)
        print(f"\n🔀 Top 5 MMR Results (diverse):")
        for i, doc in enumerate(docs_mmr, 1):
            print(f"{i}. {doc.metadata['name']} ({doc.metadata['test_type']})")
        
        # Distribution
        dist = retriever.get_type_distribution(docs_mmr)
        print(f"\n📈 Type Distribution: {dist}")


if __name__ == "__main__":
    main()
