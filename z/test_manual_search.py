#!/usr/bin/env python3
"""
Test script for the enhanced manual search functionality
"""

import requests
import json
from typing import Dict, Any


def test_manual_search_api():
    """Test the manual search API with the provided request body"""

    # API endpoint - adjust the port/host as needed
    url = "http://localhost:8000/manualsearch/"

    # Your provided request body
    request_body = {
        "experience_titles": ["Software Engineer", "Python Developer"],
        "skills": ["Python", "React", "AWS"],
        "min_education": ["BTech", "BSc"],
        "min_experience": "2 years 6 months",
        "max_experience": "5 years",
        "locations": ["Mumbai", "Pune", "Bangalore"],
        "min_salary": 500000,
        "max_salary": 1500000,
        "limit": 10,
    }

    try:
        # Make the API request
        response = requests.post(url, json=request_body, timeout=30)

        if response.status_code == 200:
            results = response.json()
            print(f"✅ API Request Successful!")
            print(f"📊 Found {len(results)} candidates")

            if results:
                print("\n🏆 Top 3 Candidates:")
                for i, candidate in enumerate(results[:3], 1):
                    print(f"\n--- Candidate {i} ---")
                    print(
                        f"Name: {candidate.get('contact_details', {}).get('name', 'N/A')}"
                    )
                    print(f"Match Score: {candidate.get('match_score', 0)}")
                    print(f"Base Score: {candidate.get('base_score', 0)}")
                    print(f"Field Match Bonus: {candidate.get('field_match_bonus', 0)}")
                    print(
                        f"Fields Matched: {candidate.get('match_details', {}).get('fields_matched', 0)}"
                    )
                    print(
                        f"Total Individual Matches: {candidate.get('total_individual_matches', 0)}"
                    )

                    match_details = candidate.get("match_details", {})
                    print(f"📋 Match Details:")
                    print(
                        f"  - Experience Titles: {match_details.get('matched_experience_titles', [])}"
                    )
                    print(f"  - Skills: {match_details.get('matched_skills', [])}")
                    print(
                        f"  - Education: {match_details.get('matched_education', [])}"
                    )
                    print(
                        f"  - Locations: {match_details.get('matched_locations', [])}"
                    )
                    print(
                        f"  - Experience Range Match: {match_details.get('experience_range_match', False)}"
                    )
                    print(
                        f"  - Salary Range Match: {match_details.get('salary_range_match', False)}"
                    )
            else:
                print("ℹ️ No candidates found matching the criteria")

        else:
            print(f"❌ API Request Failed!")
            print(f"Status Code: {response.status_code}")
            print(f"Error: {response.text}")

    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the API server is running")
    except requests.exceptions.Timeout:
        print("❌ Request Timeout: The request took too long")
    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")


def test_limit_optional():
    """Test that the limit parameter is optional"""

    url = "http://localhost:8000/manualsearch/"

    # Request without limit parameter
    request_body = {"skills": ["Python"]}

    try:
        response = requests.post(url, json=request_body, timeout=30)

        if response.status_code == 200:
            results = response.json()
            print(f"✅ Limit Optional Test Passed!")
            print(f"📊 Found {len(results)} candidates without limit parameter")
        else:
            print(f"❌ Limit Optional Test Failed!")
            print(f"Status Code: {response.status_code}")
            print(f"Error: {response.text}")

    except Exception as e:
        print(f"❌ Limit Optional Test Error: {str(e)}")


if __name__ == "__main__":
    print("🚀 Testing Enhanced Manual Search API")
    print("=" * 50)

    print("\n1. Testing with full request body:")
    test_manual_search_api()

    print("\n" + "=" * 50)
    print("\n2. Testing limit parameter is optional:")
    test_limit_optional()

    print("\n" + "=" * 50)
    print("✨ Test completed!")
