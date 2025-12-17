"""
Generate embeddings for SHL assessment catalog
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import pandas as pd
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AssessmentEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Model options:
        - all-MiniLM-L6-v2: Fast, good quality (default)
        - all-mpnet-base-v2: Better quality, slower
        - multi-qa-MiniLM-L6-cos-v1: Optimized for Q&A
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("✅ Embedding model loaded")
    
    def create_document_text(self, row: pd.Series) -> str:
        """
        Create a rich document representation from assessment data.
        This format optimizes semantic search quality.
        
        Format:
        Assessment: {name}
        Category: {category}
        Test Type: {test_type_full_name}
        Description: {description}
        Duration: {duration}
        """
        
        # Map test types to full names for better context
        test_type_map = {
            'P': 'Personality and Behavioral Assessment',
            'C': 'Cognitive and Aptitude Test',
            'K': 'Knowledge and Technical Skills',
            'S': 'Situational Judgment Test'
        }
        
        test_type_full = test_type_map.get(row['test_type'], row['test_type'])
        
        doc_text = f"""Assessment: {row['name']}
Category: {row['category']}
Test Type: {test_type_full}
Description: {row['description']}
Duration: {row.get('duration', 'Varies')}"""
        
        return doc_text.strip()
    
    def embed_catalog(self, df: pd.DataFrame) -> Tuple[List[Document], List[dict]]:
        """
        Generate embeddings for all assessments in the catalog.
        
        Returns:
            Tuple of (documents, metadatas) for vector store creation
        """
        logger.info(f"Creating embeddings for {len(df)} assessments...")
        
        documents = []
        metadatas = []
        
        for idx, row in df.iterrows():
            # Create rich document text
            doc_text = self.create_document_text(row)
            
            # Create Document object
            doc = Document(
                page_content=doc_text,
                metadata={
                    'name': row['name'],
                    'url': row['url'],
                    'test_type': row['test_type'],
                    'category': row['category'],
                    'duration': row.get('duration', 'Varies'),
                    'index': idx
                }
            )
            
            documents.append(doc)
            metadatas.append(doc.metadata)
        
        logger.info(f"✅ Created {len(documents)} document embeddings")
        
        return documents, metadatas


def main():
    """Test the embedding creation"""
    # Load catalog
    df = pd.read_csv('data/raw/shl_assessments.csv')
    logger.info(f"Loaded {len(df)} assessments from catalog")
    
    # Create embeddings
    embedder = AssessmentEmbedder()
    documents, metadatas = embedder.embed_catalog(df)
    
    # Print sample
    print("\n" + "="*80)
    print("SAMPLE DOCUMENT")
    print("="*80)
    print(documents[0].page_content)
    print("\nMetadata:")
    print(documents[0].metadata)
    print("="*80)
    
    return documents, metadatas


if __name__ == "__main__":
    documents, metadatas = main()
