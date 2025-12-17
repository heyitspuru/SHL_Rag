"""
Verify API Response Format Against Submission Requirements
"""

import requests
import json

API_URL = "http://localhost:8000"

def check_api_format():
    """Verify API response matches submission requirements"""
    
    print("="*70)
    print("API RESPONSE FORMAT VERIFICATION")
    print("="*70)
    
    # Test query from Appendix 1
    test_query = "I am hiring for Java developers who can also collaborate effectively with my business teams."
    
    print(f"\nTest Query: {test_query[:60]}...")
    print("\nSending POST request to /recommend...")
    
    payload = {
        "query": test_query,
        "top_k": 10,
        "enable_balancing": True
    }
    
    try:
        response = requests.post(
            f"{API_URL}/recommend",
            json=payload,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n" + "="*70)
            print("RESPONSE STRUCTURE CHECK")
            print("="*70)
            
            # Check required fields
            checks = {
                "Has 'query' field": "query" in result,
                "Has 'recommendations' field": "recommendations" in result,
                "Has 'total_found' field": "total_found" in result,
                "Recommendations is list": isinstance(result.get("recommendations"), list),
                "Returns 1-10 items": 1 <= len(result.get("recommendations", [])) <= 10,
            }
            
            for check, passed in checks.items():
                status = "✅" if passed else "❌"
                print(f"{status} {check}")
            
            if result.get("recommendations"):
                rec = result["recommendations"][0]
                
                print("\n" + "="*70)
                print("RECOMMENDATION FIELDS CHECK")
                print("="*70)
                
                required_fields = [
                    "assessment_name",
                    "assessment_url",
                    "test_type",
                    "category",
                    "duration_minutes",
                    "description"
                ]
                
                for field in required_fields:
                    present = field in rec
                    status = "✅" if present else "❌"
                    value = str(rec.get(field, "MISSING"))[:50]
                    print(f"{status} {field}: {value}")
                
                print("\n" + "="*70)
                print("SAMPLE RESPONSE")
                print("="*70)
                print(json.dumps({
                    "query": result["query"][:60] + "...",
                    "total_found": result["total_found"],
                    "recommendations": [
                        {
                            "assessment_name": rec["assessment_name"],
                            "assessment_url": rec["assessment_url"][:50] + "...",
                            "test_type": rec["test_type"],
                            "category": rec["category"],
                            "duration_minutes": rec.get("duration_minutes"),
                        }
                        for rec in result["recommendations"][:2]
                    ]
                }, indent=2))
                
                print("\n" + "="*70)
                print("VALIDATION RESULT")
                print("="*70)
                
                all_checks = all(checks.values())
                all_fields = all(field in rec for field in required_fields)
                
                if all_checks and all_fields:
                    print("✅ API RESPONSE FORMAT: VALID")
                    print("   Response structure matches submission requirements")
                else:
                    print("❌ API RESPONSE FORMAT: INVALID")
                    if not all_checks:
                        print("   Missing required response fields")
                    if not all_fields:
                        print("   Missing required recommendation fields")
            
        else:
            print(f"❌ API ERROR: Status {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ API NOT RUNNING")
        print("   Start API with: uvicorn src.api.main:app --reload")
    except Exception as e:
        print(f"❌ ERROR: {e}")


def test_sample_queries():
    """Test all sample queries from Appendix 1"""
    
    print("\n" + "="*70)
    print("TESTING SAMPLE QUERIES (Appendix 1)")
    print("="*70)
    
    sample_queries = [
        "I am hiring for Java developers who can also collaborate effectively with my business teams.",
        "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script.",
        "I am hiring for an analyst and wants applications to screen using Cognitive and personality tests"
    ]
    
    for i, query in enumerate(sample_queries, 1):
        print(f"\n[Query {i}]: {query[:60]}...")
        
        try:
            response = requests.post(
                f"{API_URL}/recommend",
                json={"query": query, "top_k": 10},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                count = len(result.get("recommendations", []))
                print(f"   ✅ Status: 200 OK")
                print(f"   ✅ Recommendations: {count}")
                
                # Show first recommendation
                if result.get("recommendations"):
                    first = result["recommendations"][0]
                    print(f"   Top Result: {first['assessment_name']}")
                    print(f"   Test Type: {first['test_type']}")
            else:
                print(f"   ❌ Status: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")


def main():
    print("\n" + "="*70)
    print("SHL SUBMISSION REQUIREMENTS - API VERIFICATION")
    print("="*70)
    
    # Check health first
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print("\n✅ API is running")
        else:
            print(f"\n⚠ API health check returned: {response.status_code}")
    except:
        print("\n❌ API is not running!")
        print("   Start with: uvicorn src.api.main:app --reload")
        return
    
    # Check response format
    check_api_format()
    
    # Test sample queries
    test_sample_queries()
    
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)
    print("\nNext Steps:")
    print("1. If all checks pass ✅, API is ready for submission")
    print("2. Deploy API and note the production URL")
    print("3. Test production URL with same script")
    print("4. Submit URL in evaluation form")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
