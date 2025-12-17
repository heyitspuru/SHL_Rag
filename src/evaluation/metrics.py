"""
Evaluation metrics for SHL Assessment Recommendation System
"""

import pandas as pd
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def recall_at_k(predicted_urls: List[str], true_urls: List[str], k: int = 10) -> float:
    """
    Calculate Recall@K metric.
    
    Recall@K = (Number of relevant items in top-K) / (Total number of relevant items)
    
    Args:
        predicted_urls: List of predicted assessment URLs (top-K predictions)
        true_urls: List of ground truth URLs
        k: Number of top predictions to consider
    
    Returns:
        Recall score between 0 and 1
    """
    if not true_urls:
        return 0.0
    
    # Take only top-k predictions
    top_k_predictions = set(predicted_urls[:k])
    true_set = set(true_urls)
    
    # Count correct predictions in top-k
    correct = len(top_k_predictions.intersection(true_set))
    
    # Recall = correct / total relevant
    recall = correct / len(true_set)
    
    return recall


def parse_assessment_urls(url_string: str) -> List[str]:
    """
    Parse assessment URLs from various formats.
    Handles comma-separated, pipe-separated, or newline-separated URLs.
    
    Args:
        url_string: String containing one or more URLs
    
    Returns:
        List of individual URLs
    """
    if pd.isna(url_string):
        return []
    
    url_string = str(url_string).strip()
    
    # Try different separators
    if '|' in url_string:
        urls = url_string.split('|')
    elif ',' in url_string:
        urls = url_string.split(',')
    elif '\n' in url_string:
        urls = url_string.split('\n')
    else:
        # Single URL or space-separated
        urls = [url_string]
    
    # Clean up URLs
    urls = [url.strip() for url in urls if url.strip()]
    
    return urls


def mean_recall_at_k(predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame, k: int = 10) -> float:
    """
    Calculate Mean Recall@K across all queries.
    
    Args:
        predictions_df: DataFrame with columns ['Query', 'Assessment_url']
        ground_truth_df: DataFrame with columns ['Query', 'Assessment_url']
        k: Number of top predictions to consider
    
    Returns:
        Mean recall score
    """
    # Group predictions by query
    pred_by_query = predictions_df.groupby('Query')['Assessment_url'].apply(list).to_dict()
    
    # Parse ground truth URLs (may be in different format)
    if 'Assessment_url' in ground_truth_df.columns:
        # If already parsed
        true_by_query = ground_truth_df.groupby('Query')['Assessment_url'].apply(list).to_dict()
    else:
        # Need to parse from single column
        true_by_query = {}
        for _, row in ground_truth_df.iterrows():
            query = row['Query']
            urls = parse_assessment_urls(row.get('Assessment_url', ''))
            if query not in true_by_query:
                true_by_query[query] = []
            true_by_query[query].extend(urls)
    
    # Calculate recall for each query
    recall_scores = []
    for query in true_by_query.keys():
        if query not in pred_by_query:
            logger.warning(f"Query not found in predictions: {query[:50]}...")
            recall_scores.append(0.0)
            continue
        
        predicted = pred_by_query[query]
        true_urls = true_by_query[query]
        
        recall = recall_at_k(predicted, true_urls, k=k)
        recall_scores.append(recall)
    
    if not recall_scores:
        return 0.0
    
    mean_recall = sum(recall_scores) / len(recall_scores)
    
    logger.info(f"Mean Recall@{k}: {mean_recall:.4f} (across {len(recall_scores)} queries)")
    
    return mean_recall


def evaluate_balance(predictions_df: pd.DataFrame, catalog_df: pd.DataFrame) -> Dict:
    """
    Evaluate test type balance in recommendations.
    
    Args:
        predictions_df: DataFrame with predictions
        catalog_df: Full catalog with test types
    
    Returns:
        Dictionary with balance metrics
    """
    # Create URL to test_type mapping
    url_to_type = dict(zip(catalog_df['url'], catalog_df['test_type']))
    
    # Add test types to predictions
    predictions_df = predictions_df.copy()
    predictions_df['test_type'] = predictions_df['Assessment_url'].map(url_to_type)
    
    # Overall distribution
    type_counts = predictions_df['test_type'].value_counts()
    total = len(predictions_df)
    
    type_distribution = {
        test_type: count / total 
        for test_type, count in type_counts.items()
    }
    
    # Per-query balance
    query_balances = []
    for query, group in predictions_df.groupby('Query'):
        query_dist = group['test_type'].value_counts()
        query_balances.append(len(query_dist))  # Number of different types
    
    avg_types_per_query = sum(query_balances) / len(query_balances) if query_balances else 0
    
    logger.info(f"Average test types per query: {avg_types_per_query:.2f}")
    logger.info(f"Overall type distribution: {type_distribution}")
    
    return {
        'type_distribution': type_distribution,
        'avg_types_per_query': avg_types_per_query,
        'total_recommendations': total
    }


def evaluate_model(predictions_df: pd.DataFrame, train_df: pd.DataFrame, 
                  catalog_df: pd.DataFrame, k_values: List[int] = [5, 10, 20]) -> Dict:
    """
    Comprehensive model evaluation.
    
    Args:
        predictions_df: Predictions DataFrame
        train_df: Ground truth training data
        catalog_df: Full catalog
        k_values: List of K values for Recall@K
    
    Returns:
        Dictionary with all metrics
    """
    logger.info("="*80)
    logger.info("EVALUATION RESULTS")
    logger.info("="*80)
    
    results = {}
    
    # Recall@K for different K values
    for k in k_values:
        recall = mean_recall_at_k(predictions_df, train_df, k=k)
        results[f'recall@{k}'] = recall
        logger.info(f"Recall@{k}: {recall:.4f}")
    
    # Balance evaluation
    balance_metrics = evaluate_balance(predictions_df, catalog_df)
    results['balance'] = balance_metrics
    
    logger.info(f"\nTest Type Distribution:")
    for test_type, proportion in balance_metrics['type_distribution'].items():
        logger.info(f"  {test_type}: {proportion:.2%}")
    
    logger.info(f"\nAverage types per query: {balance_metrics['avg_types_per_query']:.2f}")
    logger.info("="*80)
    
    return results


def main():
    """Test evaluation metrics"""
    # Example usage
    predictions = pd.DataFrame({
        'Query': ['test query 1'] * 10 + ['test query 2'] * 10,
        'Assessment_url': [
            'https://example.com/assess1',
            'https://example.com/assess2',
            'https://example.com/assess3',
        ] * 6 + ['https://example.com/assess4'] * 2
    })
    
    ground_truth = pd.DataFrame({
        'Query': ['test query 1', 'test query 2'],
        'Assessment_url': [
            'https://example.com/assess1,https://example.com/assess2',
            'https://example.com/assess3,https://example.com/assess4'
        ]
    })
    
    # Load catalog
    try:
        catalog = pd.read_csv('data/raw/shl_assessments.csv')
        
        # Calculate recall
        recall = mean_recall_at_k(predictions, ground_truth, k=10)
        print(f"\nMean Recall@10: {recall:.4f}")
        
        # Evaluate balance
        balance = evaluate_balance(predictions, catalog)
        print(f"\nBalance metrics: {balance}")
        
    except FileNotFoundError:
        print("Catalog file not found. Run scraper first.")


if __name__ == "__main__":
    main()
