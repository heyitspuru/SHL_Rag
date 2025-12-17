"""
SHL Assessment Recommendation Engine
Main RAG system combining retrieval, re-ranking, and balancing
"""

import os
from typing import List, Dict
import logging
import pandas as pd
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SHLRecommendationEngine:
    def __init__(self, retriever, llm=None, catalog_df=None):
        """
        Initialize the recommendation engine.
        
        Args:
            retriever: AssessmentRetriever instance
            llm: Language model for re-ranking (optional)
            catalog_df: Full catalog dataframe for metadata lookups
        """
        self.retriever = retriever
        self.llm = llm
        self.catalog_df = catalog_df
        
        logger.info("✅ Recommendation engine initialized")
    
    def analyze_query_requirements(self, query: str) -> Dict[str, float]:
        """
        Analyze query to determine required test type distribution.
        Returns dict with test_type -> proportion mapping.
        
        Uses keyword matching and LLM analysis if available.
        """
        query_lower = query.lower()
        requirements = {'K': 0.0, 'C': 0.0, 'P': 0.0, 'S': 0.0}
        
        # Technical skills indicators (Knowledge)
        if any(word in query_lower for word in [
            'java', 'python', 'sql', 'programming', 'developer', 'software',
            'technical', 'coding', 'data', 'analyst', 'engineer', 'cloud',
            'javascript', 'c++', 'machine learning', 'aws', 'azure'
        ]):
            requirements['K'] += 0.5
        
        # Cognitive/Aptitude indicators
        if any(word in query_lower for word in [
            'analytical', 'reasoning', 'problem-solving', 'critical thinking',
            'numerical', 'verbal', 'logical', 'aptitude', 'graduate'
        ]):
            requirements['C'] += 0.3
        
        # Personality/Behavioral indicators
        if any(word in query_lower for word in [
            'collaboration', 'teamwork', 'communication', 'leadership',
            'personality', 'motivated', 'traits', 'behavior', 'culture fit',
            'interpersonal', 'work style'
        ]):
            requirements['P'] += 0.3
        
        # Situational Judgment indicators
        if any(word in query_lower for word in [
            'decision-making', 'judgment', 'scenarios', 'situational',
            'customer service', 'sales', 'management'
        ]):
            requirements['S'] += 0.2
        
        # Normalize to sum to 1.0
        total = sum(requirements.values())
        if total > 0:
            requirements = {k: v/total for k, v in requirements.items()}
        else:
            # Default distribution if no keywords matched
            requirements = {'K': 0.4, 'C': 0.3, 'P': 0.2, 'S': 0.1}
        
        logger.info(f"Query requirements: {requirements}")
        return requirements
    
    def balance_recommendations(self, docs: List[Document], requirements: Dict[str, float], 
                               final_k: int = 10) -> List[Document]:
        """
        Balance recommendations according to required test type distribution.
        
        Args:
            docs: Retrieved documents
            requirements: Test type requirements (proportion dict)
            final_k: Final number of recommendations
        
        Returns:
            Balanced list of documents
        """
        # Group by test type
        by_type = {}
        for doc in docs:
            test_type = doc.metadata.get('test_type', 'K')
            if test_type not in by_type:
                by_type[test_type] = []
            by_type[test_type].append(doc)
        
        # Calculate target counts
        targets = {}
        for test_type, proportion in requirements.items():
            target = max(1, int(proportion * final_k))  # At least 1 of each required type
            targets[test_type] = min(target, len(by_type.get(test_type, [])))
        
        # Adjust targets to match final_k
        current_total = sum(targets.values())
        if current_total < final_k:
            # Add more from largest available pool
            for test_type in sorted(by_type.keys(), key=lambda t: len(by_type[t]), reverse=True):
                if current_total >= final_k:
                    break
                available = len(by_type[test_type]) - targets.get(test_type, 0)
                if available > 0:
                    add = min(available, final_k - current_total)
                    targets[test_type] = targets.get(test_type, 0) + add
                    current_total += add
        
        # Select balanced recommendations
        balanced = []
        for test_type, count in targets.items():
            if test_type in by_type:
                balanced.extend(by_type[test_type][:count])
        
        logger.info(f"Balanced to {len(balanced)} recommendations: {targets}")
        return balanced[:final_k]
    
    def rerank_with_llm(self, query: str, docs: List[Document], top_k: int = 20) -> List[Document]:
        """
        Use LLM to re-rank retrieved documents based on relevance.
        
        Args:
            query: Original query
            docs: Retrieved documents
            top_k: Number of documents to re-rank
        
        Returns:
            Re-ranked documents
        """
        if not self.llm or len(docs) <= 1:
            return docs
        
        try:
            from langchain.prompts import PromptTemplate
            
            # Prepare assessment list
            assessment_list = ""
            for i, doc in enumerate(docs[:top_k], 1):
                assessment_list += f"\n{i}. {doc.metadata['name']}\n"
                assessment_list += f"   Type: {doc.metadata['test_type']}, Category: {doc.metadata['category']}\n"
                assessment_list += f"   Description: {doc.page_content[:200]}...\n"
            
            prompt = PromptTemplate(
                input_variables=["query", "assessments"],
                template="""You are an expert in assessment selection. Given a job query and a list of assessments, rank them by relevance.

Job Query: {query}

Available Assessments:
{assessments}

Rank these assessments by relevance (most relevant first). Return ONLY the numbers in order, comma-separated (e.g., "3,1,5,2,4...").

Ranking:"""
            )
            
            response = self.llm.invoke(prompt.format(query=query, assessments=assessment_list))
            
            if hasattr(response, 'content'):
                response = response.content
            
            # Parse ranking
            ranking_str = response.strip().split('\n')[0]
            rankings = [int(x.strip()) - 1 for x in ranking_str.split(',') if x.strip().isdigit()]
            
            # Re-order documents
            reranked = [docs[i] for i in rankings if 0 <= i < len(docs)]
            # Add any missing documents
            reranked.extend([doc for i, doc in enumerate(docs) if i not in rankings])
            
            logger.info(f"LLM re-ranked {len(reranked)} documents")
            return reranked
            
        except Exception as e:
            logger.warning(f"Re-ranking failed: {e}, using original order")
            return docs
    
    def recommend(self, query: str, top_k: int = 10, enable_balancing: bool = True,
                 enable_reranking: bool = True, retrieval_k: int = 40) -> List[Dict]:
        """
        Main recommendation method.
        
        Args:
            query: Job description or natural language query
            top_k: Number of final recommendations (max 10)
            enable_balancing: Whether to balance test types
            enable_reranking: Whether to use LLM re-ranking
            retrieval_k: Initial retrieval count
        
        Returns:
            List of recommended assessments with metadata
        """
        logger.info(f"Processing recommendation request: '{query[:100]}...'")
        
        # Step 1: Retrieve candidates using MMR for diversity
        docs = self.retriever.retrieve_mmr(
            query, 
            k=retrieval_k, 
            fetch_k=retrieval_k * 2,
            lambda_mult=0.7
        )
        logger.info(f"Retrieved {len(docs)} candidates")
        
        # Step 2: LLM re-ranking (optional)
        if enable_reranking and self.llm:
            docs = self.rerank_with_llm(query, docs, top_k=min(20, len(docs)))
        
        # Step 3: Balance test types (optional)
        if enable_balancing:
            requirements = self.analyze_query_requirements(query)
            docs = self.balance_recommendations(docs, requirements, final_k=top_k)
        else:
            docs = docs[:top_k]
        
        # Step 4: Format results
        recommendations = []
        for doc in docs:
            # Extract duration in minutes (convert "30 minutes" to 30)
            duration_str = doc.metadata.get('duration', '30 minutes')
            try:
                duration_minutes = int(duration_str.split()[0]) if duration_str else 30
            except:
                duration_minutes = 30
            
            recommendations.append({
                'assessment_name': doc.metadata['name'],
                'assessment_url': doc.metadata['url'],
                'test_type': doc.metadata['test_type'],
                'category': doc.metadata['category'],
                'duration_minutes': duration_minutes,
                'description': doc.page_content if doc.page_content else doc.metadata.get('description', '')
            })
        
        logger.info(f"✅ Returning {len(recommendations)} recommendations")
        return recommendations
    
    def batch_recommend(self, queries: List[str], top_k: int = 10) -> pd.DataFrame:
        """
        Generate recommendations for multiple queries.
        Returns DataFrame with columns: Query, Assessment_url
        """
        results = []
        
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}")
            recommendations = self.recommend(query, top_k=top_k)
            
            for rec in recommendations:
                results.append({
                    'Query': query,
                    'Assessment_url': rec['assessment_url']
                })
        
        return pd.DataFrame(results)


def main():
    """Test the recommendation engine"""
    import sys
    sys.path.append('src')
    
    from rag.embeddings import AssessmentEmbedder
    from rag.vectorstore import VectorStoreManager
    from rag.retriever import AssessmentRetriever
    
    # Initialize components
    logger.info("Initializing recommendation system...")
    
    # Load catalog
    catalog_df = pd.read_csv('data/raw/shl_assessments.csv')
    
    # Load embeddings and vector store
    embedder = AssessmentEmbedder()
    vs_manager = VectorStoreManager(embedder.embeddings)
    
    try:
        vectorstore = vs_manager.load_vectorstore(use_chroma=True)
    except:
        logger.info("Building new vector store...")
        documents, _ = embedder.embed_catalog(catalog_df)
        vectorstore = vs_manager.create_vectorstore(documents, use_chroma=True)
    
    # Create retriever
    retriever = AssessmentRetriever(vectorstore)
    
    # Create recommendation engine (without LLM for now)
    engine = SHLRecommendationEngine(
        retriever=retriever,
        llm=None,  # Can add Gemini here
        catalog_df=catalog_df
    )
    
    # Test queries
    test_queries = [
        "I am hiring for Java developers who can also collaborate well in teams",
        "Looking for entry-level sales representatives with customer service skills",
        "Need data analysts with strong numerical and analytical abilities"
    ]
    
    for query in test_queries:
        print("\n" + "="*80)
        print(f"QUERY: {query}")
        print("="*80)
        
        recommendations = engine.recommend(
            query,
            top_k=10,
            enable_balancing=True,
            enable_reranking=False
        )
        
        print(f"\n📋 Top {len(recommendations)} Recommendations:\n")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['assessment_name']}")
            print(f"   Type: {rec['test_type']} | Category: {rec['category']} | Duration: {rec['duration']}")
            print(f"   URL: {rec['assessment_url']}")
            print()
        
        # Show type distribution
        type_dist = {}
        for rec in recommendations:
            t = rec['test_type']
            type_dist[t] = type_dist.get(t, 0) + 1
        print(f"📊 Type Distribution: {type_dist}")
        print("="*80)


if __name__ == "__main__":
    main()
