"""
Generate predictions on test set for submission
Output format: Query, Assessment_url (one row per recommendation)
"""

import sys
import os
import pandas as pd
import logging

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag.embeddings import AssessmentEmbedder
from rag.vectorstore import VectorStoreManager
from rag.retriever import AssessmentRetriever
from rag.recommender import SHLRecommendationEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_test_predictions(
    test_csv_path: str = 'data/raw/test.csv',
    output_path: str = 'data/predictions/test_predictions.csv',
    top_k: int = 10,
    enable_balancing: bool = True
):
    """
    Generate predictions on test set.
    
    Args:
        test_csv_path: Path to test CSV with 'Query' column
        output_path: Path to save predictions
        top_k: Number of recommendations per query
        enable_balancing: Whether to balance test types
    
    Returns:
        DataFrame with predictions
    """
    logger.info("="*80)
    logger.info("GENERATING TEST PREDICTIONS")
    logger.info("="*80)
    
    # Load test data
    logger.info(f"\n1. Loading test data from {test_csv_path}...")
    test_df = pd.read_csv(test_csv_path)
    logger.info(f"   ✅ Loaded {len(test_df)} test queries")
    
    # Load catalog
    logger.info("\n2. Loading assessment catalog...")
    catalog_df = pd.read_csv('data/raw/shl_assessments.csv')
    logger.info(f"   ✅ Loaded {len(catalog_df)} assessments")
    
    # Initialize recommendation engine
    logger.info("\n3. Initializing recommendation engine...")
    
    # Load embeddings
    embedder = AssessmentEmbedder(model_name="all-MiniLM-L6-v2")
    
    # Load vector store
    vs_manager = VectorStoreManager(embedder.embeddings)
    try:
        vectorstore = vs_manager.load_vectorstore(use_chroma=True)
        logger.info("   ✅ Vector store loaded")
    except Exception as e:
        logger.error(f"   ❌ Failed to load vector store: {e}")
        logger.info("   Building new vector store...")
        documents, _ = embedder.embed_catalog(catalog_df)
        vectorstore = vs_manager.create_vectorstore(documents, use_chroma=True)
    
    # Create retriever and engine
    retriever = AssessmentRetriever(vectorstore)
    engine = SHLRecommendationEngine(
        retriever=retriever,
        llm=None,
        catalog_df=catalog_df
    )
    logger.info("   ✅ Recommendation engine ready")
    
    # Generate predictions
    logger.info(f"\n4. Generating predictions for {len(test_df)} queries...")
    
    all_predictions = []
    
    for i, row in test_df.iterrows():
        query = row['Query']
        logger.info(f"   Processing {i+1}/{len(test_df)}: {query[:60]}...")
        
        try:
            recommendations = engine.recommend(
                query=query,
                top_k=top_k,
                enable_balancing=enable_balancing,
                enable_reranking=False
            )
            
            # Add to predictions
            for rec in recommendations:
                all_predictions.append({
                    'Query': query,
                    'Assessment_url': rec['assessment_url']
                })
        
        except Exception as e:
            logger.error(f"   ❌ Failed for query {i+1}: {e}")
            # Add empty predictions to maintain structure
            for _ in range(top_k):
                all_predictions.append({
                    'Query': query,
                    'Assessment_url': ''
                })
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame(all_predictions)
    
    # Save predictions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    logger.info(f"\n5. ✅ Predictions saved to: {output_path}")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("PREDICTION SUMMARY")
    logger.info("="*80)
    logger.info(f"Total predictions: {len(predictions_df)}")
    logger.info(f"Queries processed: {len(test_df)}")
    logger.info(f"Predictions per query: {top_k}")
    logger.info(f"Output file: {output_path}")
    logger.info("="*80)
    
    return predictions_df


def validate_predictions(predictions_df: pd.DataFrame):
    """Validate prediction format"""
    logger.info("\n" + "="*80)
    logger.info("VALIDATING PREDICTIONS")
    logger.info("="*80)
    
    # Check columns
    required_cols = ['Query', 'Assessment_url']
    for col in required_cols:
        if col not in predictions_df.columns:
            logger.error(f"❌ Missing required column: {col}")
            return False
        else:
            logger.info(f"✅ Column '{col}' present")
    
    # Check for empty values
    empty_queries = predictions_df['Query'].isna().sum()
    empty_urls = predictions_df['Assessment_url'].isna().sum()
    
    logger.info(f"\nEmpty values:")
    logger.info(f"  Queries: {empty_queries}")
    logger.info(f"  URLs: {empty_urls}")
    
    # Count predictions per query
    counts = predictions_df.groupby('Query').size()
    logger.info(f"\nPredictions per query:")
    logger.info(f"  Min: {counts.min()}")
    logger.info(f"  Max: {counts.max()}")
    logger.info(f"  Mean: {counts.mean():.2f}")
    
    logger.info("\n✅ Validation complete!")
    logger.info("="*80)
    
    return True


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate test predictions')
    parser.add_argument('--test-file', default='data/raw/test.csv',
                       help='Path to test CSV file')
    parser.add_argument('--output', default='data/predictions/test_predictions.csv',
                       help='Path to save predictions')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of recommendations per query')
    parser.add_argument('--no-balancing', action='store_true',
                       help='Disable test type balancing')
    
    args = parser.parse_args()
    
    # Generate predictions
    predictions_df = generate_test_predictions(
        test_csv_path=args.test_file,
        output_path=args.output,
        top_k=args.top_k,
        enable_balancing=not args.no_balancing
    )
    
    # Validate
    validate_predictions(predictions_df)
    
    logger.info("\n✅ ALL DONE! Predictions are ready for submission.")


if __name__ == "__main__":
    main()
