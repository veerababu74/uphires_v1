#!/usr/bin/env python3
"""
Test script for manual search save/recent functionality
"""

import asyncio
import httpx
import json

# Test payload as provided by the user
test_payload = {
    "userid": "test_user_123",
    "experience_titles": ["Software Engineer", "Python Developer"],
    "skills": ["Python", "React", "AWS"],
    "min_education": ["BTech", "BSc"],
    "min_experience": "2 years 6 months",
    "max_experience": "5 years",
    "locations": ["Mumbai", "Pune", "Bangalore"],
    "min_salary": 500000,
    "max_salary": 1500000,
}

BASE_URL = "http://localhost:8000"


async def test_manual_search_operations():
    """Test all manual search save/recent operations"""

    async with httpx.AsyncClient() as client:

        print("🔍 Testing Manual Search Save/Recent Operations")
        print("=" * 50)

        # Test 1: Save search to saved searches
        print("\n1. Testing Save Search to Saved Collection...")
        try:
            response = await client.post(
                f"{BASE_URL}/manual_search_operations/save_search", json=test_payload
            )
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success: {result['message']}")
                print(f"Search ID: {result['search_id']}")
                print(f"Timestamp: {result['timestamp']}")
                saved_search_id = result["search_id"]
            else:
                print(f"❌ Error: {response.text}")
                saved_search_id = None
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
            saved_search_id = None

        # Test 2: Save search to recent searches
        print("\n2. Testing Save Search to Recent Collection...")
        try:
            response = await client.post(
                f"{BASE_URL}/manual_search_operations/save_recent_search",
                json=test_payload,
            )
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success: {result['message']}")
                print(f"Search ID: {result['search_id']}")
                print(f"Timestamp: {result['timestamp']}")
                recent_search_id = result["search_id"]
            else:
                print(f"❌ Error: {response.text}")
                recent_search_id = None
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
            recent_search_id = None  # Test 3: Get saved searches for user (with limit)
        print("\n3. Testing Get Saved Searches with Limit...")
        try:
            # Test with limit=5
            response = await client.get(
                f"{BASE_URL}/manual_search_operations/saved_searches/{test_payload['userid']}?limit=5"
            )
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(
                    f"✅ Success: Found {result['total_count']} saved searches (limit=5)"
                )
                if result["searches"]:
                    print("Recent saved search:")
                    latest_search = result["searches"][0]
                    print(f"  - Search ID: {latest_search['search_id']}")
                    print(f"  - Timestamp: {latest_search['timestamp']}")
                    print(f"  - Skills: {latest_search['search_criteria']['skills']}")
            else:
                print(f"❌ Error: {response.text}")
        except Exception as e:
            print(f"❌ Exception: {str(e)}")

        # Test 4: Get recent searches for user (with limit)
        print("\n4. Testing Get Recent Searches with Custom Limit...")
        try:
            # Test with limit=3
            response = await client.get(
                f"{BASE_URL}/manual_search_operations/recent_searches/{test_payload['userid']}?limit=3"
            )
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(
                    f"✅ Success: Found {result['total_count']} recent searches (limit=3)"
                )
                if result["searches"]:
                    print("Recent search:")
                    latest_search = result["searches"][0]
                    print(f"  - Search ID: {latest_search['search_id']}")
                    print(f"  - Timestamp: {latest_search['timestamp']}")
                    print(
                        f"  - Locations: {latest_search['search_criteria']['locations']}"
                    )
            else:
                print(f"❌ Error: {response.text}")
        except Exception as e:
            print(f"❌ Exception: {str(e)}")

        # Test 5: Get recent searches with default limit
        print("\n5. Testing Get Recent Searches with Default Limit...")
        try:
            response = await client.get(
                f"{BASE_URL}/manual_search_operations/recent_searches/{test_payload['userid']}"
            )
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(
                    f"✅ Success: Found {result['total_count']} recent searches (default limit=10)"
                )
            else:
                print(f"❌ Error: {response.text}")
        except Exception as e:
            print(f"❌ Exception: {str(e)}")

        print("\n" + "=" * 50)
        print("🎉 Manual Search Save/Recent Testing Completed!")
        print("\nAPI Endpoints Created (with limit support):")
        print("📝 POST /manual_search_operations/save_search")
        print("📝 POST /manual_search_operations/save_recent_search")
        print(
            "📋 GET /manual_search_operations/saved_searches/{user_id}?limit={optional}"
        )
        print(
            "📋 GET /manual_search_operations/recent_searches/{user_id}?limit={optional}"
        )
        print("🗑️ DELETE /manual_search_operations/saved_searches/{search_id}")
        print("🗑️ DELETE /manual_search_operations/recent_searches/{search_id}")
        print("\nLimit Parameters:")
        print("- Saved searches: No default limit, can specify any number")
        print("- Recent searches: Default limit=10, max limit=50")


def test_payload_structure():
    """Test that the payload structure is correct"""
    print("\n🧪 Testing Payload Structure")
    print("=" * 30)
    print("Payload to be saved:")
    print(json.dumps(test_payload, indent=2))

    required_fields = ["userid"]
    optional_fields = [
        "experience_titles",
        "skills",
        "min_education",
        "min_experience",
        "max_experience",
        "locations",
        "min_salary",
        "max_salary",
    ]

    print(
        f"\n✅ Required fields present: {all(field in test_payload for field in required_fields)}"
    )
    print(
        f"📋 Optional fields in payload: {[field for field in optional_fields if field in test_payload]}"
    )


if __name__ == "__main__":
    test_payload_structure()

    print("\n" + "=" * 60)
    print("🚀 To test the API endpoints, start the server first:")
    print("   uvicorn main:app --reload")
    print("Then run:")
    print("   python test_manual_search_save.py")
    print("=" * 60)

    # Uncomment the line below to run the actual API tests
    # (make sure the server is running first)
    # asyncio.run(test_manual_search_operations())
