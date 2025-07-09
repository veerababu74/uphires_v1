#!/usr/bin/env python3
"""
Comprehensive testing script for Resume Search API

This script tests all major components and endpoints to ensure
the application is working correctly after setup.

Usage:
    python test_complete_setup.py
"""

import requests
import json
import time
import sys
from typing import Dict, Any, List
import asyncio
from dataclasses import dataclass


@dataclass
class TestResult:
    name: str
    success: bool
    message: str
    response_time: float = 0.0


class ResumeAPITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[TestResult] = []

    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")

    def test_health_check(self) -> TestResult:
        """Test basic health check endpoint"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/health/health", timeout=10)
            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    return TestResult(
                        name="Health Check",
                        success=True,
                        message="Basic health check passed",
                        response_time=response_time,
                    )
                else:
                    return TestResult(
                        name="Health Check",
                        success=False,
                        message=f"Unhealthy status: {data.get('status')}",
                        response_time=response_time,
                    )
            else:
                return TestResult(
                    name="Health Check",
                    success=False,
                    message=f"HTTP {response.status_code}: {response.text}",
                    response_time=response_time,
                )
        except Exception as e:
            return TestResult(
                name="Health Check",
                success=False,
                message=f"Connection error: {str(e)}",
            )

    def test_detailed_health_check(self) -> TestResult:
        """Test detailed health check endpoint"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/health/detailed", timeout=15)
            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                components = data.get("components", {})
                database_status = components.get("database", {}).get("status")

                if database_status == "healthy":
                    return TestResult(
                        name="Detailed Health Check",
                        success=True,
                        message=f"All components healthy. DB: {database_status}",
                        response_time=response_time,
                    )
                else:
                    return TestResult(
                        name="Detailed Health Check",
                        success=False,
                        message=f"Database unhealthy: {database_status}",
                        response_time=response_time,
                    )
            else:
                return TestResult(
                    name="Detailed Health Check",
                    success=False,
                    message=f"HTTP {response.status_code}: {response.text}",
                    response_time=response_time,
                )
        except Exception as e:
            return TestResult(
                name="Detailed Health Check", success=False, message=f"Error: {str(e)}"
            )

    def test_database_health(self) -> TestResult:
        """Test database health endpoint"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/health/database", timeout=10)
            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    db_info = data.get("database", {})
                    return TestResult(
                        name="Database Health",
                        success=True,
                        message=f"Database healthy. Collection: {db_info.get('collection')}",
                        response_time=response_time,
                    )
                else:
                    return TestResult(
                        name="Database Health",
                        success=False,
                        message=f"Database status: {data.get('status')}",
                        response_time=response_time,
                    )
            else:
                return TestResult(
                    name="Database Health",
                    success=False,
                    message=f"HTTP {response.status_code}: {response.text}",
                    response_time=response_time,
                )
        except Exception as e:
            return TestResult(
                name="Database Health", success=False, message=f"Error: {str(e)}"
            )

    def test_llm_config(self) -> TestResult:
        """Test LLM configuration endpoint"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/api/llm_config", timeout=10)
            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    config_data = data.get("data", {})
                    provider = config_data.get("provider")
                    return TestResult(
                        name="LLM Configuration",
                        success=True,
                        message=f"LLM configured. Provider: {provider}",
                        response_time=response_time,
                    )
                else:
                    return TestResult(
                        name="LLM Configuration",
                        success=False,
                        message=f"LLM config failed: {data.get('error')}",
                        response_time=response_time,
                    )
            else:
                return TestResult(
                    name="LLM Configuration",
                    success=False,
                    message=f"HTTP {response.status_code}: {response.text}",
                    response_time=response_time,
                )
        except Exception as e:
            return TestResult(
                name="LLM Configuration", success=False, message=f"Error: {str(e)}"
            )

    def test_add_user_data(self) -> TestResult:
        """Test adding user data endpoint"""
        try:
            test_data = {
                "user_id": "test_user_" + str(int(time.time())),
                "contact_details": {
                    "name": "Test User",
                    "email": f"test_{int(time.time())}@example.com",
                    "phone_number": "+91-9876543210",
                    "current_city": "Bangalore",
                    "pan_card": f"TEST{int(time.time())}",
                    "addhar_number": f"12345678{int(time.time())}",
                },
                "skills": ["Python", "FastAPI", "Testing"],
                "may_also_known_skills": ["MongoDB", "Docker"],
                "total_experience": 3,
                "current_salary": 60000,
                "expected_salary": 75000,
                "notice_period": 30,
                "experience": [
                    {
                        "title": "Software Engineer",
                        "company": "Test Company",
                        "duration": "2 years",
                        "description": "Developed and tested applications",
                    }
                ],
            }

            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/add_user_data", json=test_data, timeout=15
            )
            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    return TestResult(
                        name="Add User Data",
                        success=True,
                        message=f"User data added successfully. User ID: {test_data['user_id']}",
                        response_time=response_time,
                    )
                else:
                    return TestResult(
                        name="Add User Data",
                        success=False,
                        message=f"Failed to add user: {data.get('error')}",
                        response_time=response_time,
                    )
            else:
                return TestResult(
                    name="Add User Data",
                    success=False,
                    message=f"HTTP {response.status_code}: {response.text}",
                    response_time=response_time,
                )
        except Exception as e:
            return TestResult(
                name="Add User Data", success=False, message=f"Error: {str(e)}"
            )

    def test_vector_search(self) -> TestResult:
        """Test vector search endpoint"""
        try:
            search_data = {
                "query": "experienced python developer with web frameworks",
                "k": 5,
                "user_id": "test_searcher",
                "filters": {"total_experience_min": 1, "skills": ["Python"]},
            }

            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/vector_search", json=search_data, timeout=20
            )
            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    results = data.get("data", {}).get("results", [])
                    return TestResult(
                        name="Vector Search",
                        success=True,
                        message=f"Vector search completed. Found {len(results)} results",
                        response_time=response_time,
                    )
                else:
                    return TestResult(
                        name="Vector Search",
                        success=False,
                        message=f"Search failed: {data.get('error')}",
                        response_time=response_time,
                    )
            else:
                return TestResult(
                    name="Vector Search",
                    success=False,
                    message=f"HTTP {response.status_code}: {response.text}",
                    response_time=response_time,
                )
        except Exception as e:
            return TestResult(
                name="Vector Search", success=False, message=f"Error: {str(e)}"
            )

    def test_manual_search(self) -> TestResult:
        """Test manual search endpoint"""
        try:
            search_data = {
                "skills": ["Python"],
                "total_experience_min": 1,
                "total_experience_max": 10,
                "page": 1,
                "limit": 10,
            }

            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/manual_search", json=search_data, timeout=15
            )
            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    results = data.get("data", {}).get("results", [])
                    return TestResult(
                        name="Manual Search",
                        success=True,
                        message=f"Manual search completed. Found {len(results)} results",
                        response_time=response_time,
                    )
                else:
                    return TestResult(
                        name="Manual Search",
                        success=False,
                        message=f"Search failed: {data.get('error')}",
                        response_time=response_time,
                    )
            else:
                return TestResult(
                    name="Manual Search",
                    success=False,
                    message=f"HTTP {response.status_code}: {response.text}",
                    response_time=response_time,
                )
        except Exception as e:
            return TestResult(
                name="Manual Search", success=False, message=f"Error: {str(e)}"
            )

    def test_autocomplete_skills(self) -> TestResult:
        """Test skills autocomplete endpoint"""
        try:
            start_time = time.time()
            response = requests.get(
                f"{self.base_url}/api/autocomplete_skills?q=pyth&limit=5", timeout=10
            )
            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    suggestions = data.get("data", {}).get("suggestions", [])
                    return TestResult(
                        name="Skills Autocomplete",
                        success=True,
                        message=f"Autocomplete working. Found {len(suggestions)} suggestions",
                        response_time=response_time,
                    )
                else:
                    return TestResult(
                        name="Skills Autocomplete",
                        success=False,
                        message=f"Autocomplete failed: {data.get('error')}",
                        response_time=response_time,
                    )
            else:
                return TestResult(
                    name="Skills Autocomplete",
                    success=False,
                    message=f"HTTP {response.status_code}: {response.text}",
                    response_time=response_time,
                )
        except Exception as e:
            return TestResult(
                name="Skills Autocomplete", success=False, message=f"Error: {str(e)}"
            )

    def test_cities_endpoint(self) -> TestResult:
        """Test cities endpoint"""
        try:
            start_time = time.time()
            response = requests.get(
                f"{self.base_url}/api/cities?q=bang&limit=5", timeout=10
            )
            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    cities = data.get("data", {}).get("cities", [])
                    return TestResult(
                        name="Cities Endpoint",
                        success=True,
                        message=f"Cities endpoint working. Found {len(cities)} cities",
                        response_time=response_time,
                    )
                else:
                    return TestResult(
                        name="Cities Endpoint",
                        success=False,
                        message=f"Cities failed: {data.get('error')}",
                        response_time=response_time,
                    )
            else:
                return TestResult(
                    name="Cities Endpoint",
                    success=False,
                    message=f"HTTP {response.status_code}: {response.text}",
                    response_time=response_time,
                )
        except Exception as e:
            return TestResult(
                name="Cities Endpoint", success=False, message=f"Error: {str(e)}"
            )

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return results"""
        self.log("Starting comprehensive API tests...")

        # Define tests to run
        tests = [
            self.test_health_check,
            self.test_detailed_health_check,
            self.test_database_health,
            self.test_llm_config,
            self.test_add_user_data,
            self.test_vector_search,
            self.test_manual_search,
            self.test_autocomplete_skills,
            self.test_cities_endpoint,
        ]

        # Run tests
        for test_func in tests:
            self.log(f"Running {test_func.__name__}...")
            result = test_func()
            self.results.append(result)

            if result.success:
                self.log(
                    f"✅ {result.name}: {result.message} ({result.response_time:.2f}s)",
                    "SUCCESS",
                )
            else:
                self.log(f"❌ {result.name}: {result.message}", "ERROR")

            # Small delay between tests
            time.sleep(0.5)

        # Calculate summary
        successful_tests = [r for r in self.results if r.success]
        failed_tests = [r for r in self.results if not r.success]

        summary = {
            "total_tests": len(self.results),
            "successful": len(successful_tests),
            "failed": len(failed_tests),
            "success_rate": len(successful_tests) / len(self.results) * 100,
            "average_response_time": sum(r.response_time for r in self.results)
            / len(self.results),
            "results": self.results,
        }

        return summary

    def print_summary(self, summary: Dict[str, Any]):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("                TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Average Response Time: {summary['average_response_time']:.2f}s")
        print("=" * 60)

        if summary["failed"] > 0:
            print("\nFAILED TESTS:")
            for result in summary["results"]:
                if not result.success:
                    print(f"❌ {result.name}: {result.message}")

        print("\nSUCCESSFUL TESTS:")
        for result in summary["results"]:
            if result.success:
                print(f"✅ {result.name}: {result.message}")

        print("\n" + "=" * 60)


def main():
    """Main function to run tests"""
    import argparse

    parser = argparse.ArgumentParser(description="Test Resume Search API")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)",
    )
    args = parser.parse_args()

    tester = ResumeAPITester(args.url)
    tester.log(f"Testing API at: {args.url}")

    try:
        summary = tester.run_all_tests()
        tester.print_summary(summary)

        # Exit with error code if tests failed
        if summary["failed"] > 0:
            sys.exit(1)
        else:
            tester.log("All tests passed successfully! 🎉", "SUCCESS")
            sys.exit(0)

    except KeyboardInterrupt:
        tester.log("Tests interrupted by user", "WARNING")
        sys.exit(130)
    except Exception as e:
        tester.log(f"Unexpected error: {str(e)}", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()
