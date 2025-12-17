"""
Quick Test Script - Verify Production System
"""

import requests
import json

# Test configuration
API_URL = "http://localhost:8000"
TEST_QUERY = "Java programming assessment for senior developers"

def test_health():
    """Test health endpoint"""
    print("=" * 60)
    print("Testing Health Endpoint")
    print("=" * 60)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_recommendation():
    """Test recommendation endpoint"""
    print("=" * 60)
    print("Testing Recommendation Endpoint")
    print("=" * 60)
    
    payload = {
        "query": TEST_QUERY,
        "top_k": 5,
        "enable_balancing": True
    }
    
    response = requests.post(
        f"{API_URL}/recommend",
        json=payload
    )
    
    print(f"Status: {response.status_code}")
    result = response.json()
    
    print(f"\nQuery: {TEST_QUERY}")
    print(f"Found: {len(result['recommendations'])} recommendations")
    print("\nTop 5 Results:")
    
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"\n{i}. {rec['assessment_name']}")
        print(f"   Type: {rec['test_type']}")
        score = rec.get('similarity_score', rec.get('score', 0))
        if score:
            print(f"   Score: {score:.4f}")
        print(f"   URL: {rec['assessment_url'][:60]}...")

def test_stats():
    """Test stats endpoint"""
    print("\n" + "=" * 60)
    print("Testing Stats Endpoint")
    print("=" * 60)
    
    response = requests.get(f"{API_URL}/stats")
    print(f"Status: {response.status_code}")
    stats = response.json()
    
    print(f"\nTotal Assessments: {stats['total_assessments']}")
    print(f"Assessment Categories: {', '.join(stats['categories'])}")
    print("\nTest Type Distribution:")
    for test_type, count in stats['test_types'].items():
        print(f"  {test_type}: {count}")

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("SHL RAG SYSTEM - PRODUCTION TEST")
    print("=" * 60 + "\n")
    
    try:
        test_health()
        test_recommendation()
        test_stats()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - SYSTEM READY!")
        print("=" * 60)
        
        print("\nNext Steps:")
        print("1. API is running at: http://localhost:8000")
        print("2. API Docs available at: http://localhost:8000/docs")
        print("3. Start frontend: streamlit run src/frontend/app.py")
        print("4. Review DEPLOYMENT.md for cloud deployment")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)

if __name__ == "__main__":
    main()
