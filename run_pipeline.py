"""
Production Pipeline - Complete End-to-End Execution
Runs all steps from data loading to prediction generation
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def run_step(step_name, func):
    """Run a pipeline step with error handling"""
    print_header(f"STEP: {step_name}")
    try:
        func()
        logger.info(f"✅ {step_name} - SUCCESS")
        return True
    except Exception as e:
        logger.error(f"❌ {step_name} - FAILED: {e}")
        return False


def step1_verify_data():
    """Verify dataset files exist"""
    logger.info("Checking for required data files...")
    
    required_files = [
        'data/raw/shl_assessments.csv',
        'data/raw/train.csv',
        'data/raw/test.csv'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Missing required file: {file}")
        logger.info(f"  ✓ Found: {file}")
    
    # Check assessment count
    import pandas as pd
    df = pd.read_csv('data/raw/shl_assessments.csv')
    logger.info(f"  ✓ Assessment catalog: {len(df)} items")
    
    train_df = pd.read_csv('data/raw/train.csv')
    logger.info(f"  ✓ Training queries: {len(train_df)}")
    
    test_df = pd.read_csv('data/raw/test.csv')
    logger.info(f"  ✓ Test queries: {len(test_df)}")


def step2_build_vectorstore():
    """Build or verify vector store"""
    logger.info("Building vector store...")
    
    from rag.embeddings import AssessmentEmbedder
    from rag.vectorstore import VectorStoreManager
    import pandas as pd
    
    # Load catalog
    df = pd.read_csv('data/raw/shl_assessments.csv')
    logger.info(f"  Loading {len(df)} assessments...")
    
    # Create embeddings
    embedder = AssessmentEmbedder(model_name="all-MiniLM-L6-v2")
    documents, _ = embedder.embed_catalog(df)
    logger.info(f"  ✓ Created embeddings for {len(documents)} documents")
    
    # Build vector store
    vs_manager = VectorStoreManager(
        embeddings=embedder.embeddings,
        persist_directory="data/processed/vectorstore"
    )
    vectorstore = vs_manager.create_vectorstore(documents, use_chroma=True)
    logger.info(f"  ✓ Vector store built and persisted")
    
    # Test retrieval
    test_query = "Java programming"
    results = vs_manager.similarity_search(test_query, k=3)
    logger.info(f"  ✓ Test query successful: '{test_query}' → {len(results)} results")


def step3_generate_predictions():
    """Generate predictions for test set"""
    logger.info("Generating test predictions...")
    
    from rag.embeddings import AssessmentEmbedder
    from rag.vectorstore import VectorStoreManager
    from rag.retriever import AssessmentRetriever
    from rag.recommender import SHLRecommendationEngine
    import pandas as pd
    
    # Load test data
    test_df = pd.read_csv('data/raw/test.csv')
    logger.info(f"  Processing {len(test_df)} test queries...")
    
    # Load catalog
    catalog_df = pd.read_csv('data/raw/shl_assessments.csv')
    
    # Initialize system
    embedder = AssessmentEmbedder(model_name="all-MiniLM-L6-v2")
    vs_manager = VectorStoreManager(embedder.embeddings)
    vectorstore = vs_manager.load_vectorstore(use_chroma=True)
    retriever = AssessmentRetriever(vectorstore)
    engine = SHLRecommendationEngine(retriever, catalog_df=catalog_df)
    
    # Generate predictions
    all_predictions = []
    for i, row in test_df.iterrows():
        query = row['Query']
        logger.info(f"  Processing {i+1}/{len(test_df)}: {query[:60]}...")
        
        recommendations = engine.recommend(query, top_k=10, enable_balancing=True)
        
        for rec in recommendations:
            all_predictions.append({
                'Query': query,
                'Assessment_url': rec['assessment_url']
            })
    
    # Save predictions
    predictions_df = pd.DataFrame(all_predictions)
    output_path = 'data/predictions/test_predictions.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    
    logger.info(f"  ✓ Saved {len(predictions_df)} predictions to {output_path}")
    logger.info(f"  ✓ {len(test_df)} queries × 10 recommendations = {len(predictions_df)} total")


def step4_validate_output():
    """Validate prediction output format"""
    logger.info("Validating predictions...")
    
    import pandas as pd
    
    predictions_df = pd.read_csv('data/predictions/test_predictions.csv')
    
    # Check columns
    required_cols = ['Query', 'Assessment_url']
    for col in required_cols:
        if col not in predictions_df.columns:
            raise ValueError(f"Missing required column: {col}")
        logger.info(f"  ✓ Column '{col}' present")
    
    # Check for empty values
    empty_queries = predictions_df['Query'].isna().sum()
    empty_urls = predictions_df['Assessment_url'].isna().sum()
    
    if empty_queries > 0 or empty_urls > 0:
        logger.warning(f"  ⚠ Empty values found: queries={empty_queries}, urls={empty_urls}")
    else:
        logger.info(f"  ✓ No empty values")
    
    # Count predictions per query
    counts = predictions_df.groupby('Query').size()
    logger.info(f"  ✓ Predictions per query: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
    
    logger.info(f"  ✓ Total predictions: {len(predictions_df)}")


def step5_system_check():
    """Verify all components are working"""
    logger.info("Running system checks...")
    
    # Check vector store exists
    if not os.path.exists('data/processed/vectorstore'):
        raise FileNotFoundError("Vector store not found")
    logger.info("  ✓ Vector store exists")
    
    # Check predictions exist
    if not os.path.exists('data/predictions/test_predictions.csv'):
        raise FileNotFoundError("Predictions file not found")
    logger.info("  ✓ Predictions file exists")
    
    # Check all source files
    required_src = [
        'src/rag/embeddings.py',
        'src/rag/vectorstore.py',
        'src/rag/retriever.py',
        'src/rag/recommender.py',
        'src/api/main.py',
        'src/frontend/app.py'
    ]
    
    for file in required_src:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Missing source file: {file}")
    logger.info("  ✓ All source files present")
    
    logger.info("  ✓ System ready for deployment")


def main():
    """Run complete production pipeline"""
    print_header("SHL RAG PRODUCTION PIPELINE")
    
    logger.info("Starting production pipeline execution...")
    logger.info(f"Working directory: {os.getcwd()}")
    
    steps = [
        ("Verify Data Files", step1_verify_data),
        ("Build Vector Store", step2_build_vectorstore),
        ("Generate Predictions", step3_generate_predictions),
        ("Validate Output", step4_validate_output),
        ("System Check", step5_system_check),
    ]
    
    results = []
    for step_name, step_func in steps:
        success = run_step(step_name, step_func)
        results.append((step_name, success))
        if not success:
            logger.error(f"Pipeline failed at: {step_name}")
            break
    
    # Summary
    print_header("PIPELINE SUMMARY")
    for step_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}  {step_name}")
    
    all_success = all(success for _, success in results)
    
    if all_success:
        print_header("🎉 PIPELINE COMPLETE - READY FOR DEPLOYMENT")
        print("\nNext steps:")
        print("  1. Review predictions: data/predictions/test_predictions.csv")
        print("  2. Start API: uvicorn src.api.main:app --reload")
        print("  3. Start Frontend: streamlit run src/frontend/app.py")
        print("  4. Deploy to cloud (see README.md)")
        print("\n" + "="*80)
    else:
        print_header("❌ PIPELINE FAILED")
        print("Please check the errors above and fix before deployment.")
        print("="*80)
        sys.exit(1)


if __name__ == "__main__":
    main()
