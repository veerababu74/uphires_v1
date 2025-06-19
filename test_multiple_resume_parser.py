"""
Test script for the multiple resume parser endpoints.
This script demonstrates how to use the new threading and multiprocessing endpoints.
"""

import requests
import time
import os
from pathlib import Path

# API base URL - adjust this to your server
BASE_URL = "http://localhost:8000"  # Change this to your actual server URL


def test_single_resume_parser(file_path):
    """Test the single resume parser endpoint"""
    print(f"\n{'='*50}")
    print("Testing Single Resume Parser")
    print(f"{'='*50}")

    url = f"{BASE_URL}/resume-parser"

    with open(file_path, "rb") as file:
        files = {"file": (os.path.basename(file_path), file)}
        start_time = time.time()
        response = requests.post(url, files=files)
        end_time = time.time()

    print(f"Status Code: {response.status_code}")
    print(f"Processing Time: {end_time - start_time:.2f} seconds")

    if response.status_code == 200:
        result = response.json()
        print(f"File: {result['filename']}")
        print(f"Resume extracted successfully!")
    else:
        print(f"Error: {response.text}")


def test_multiple_resume_parser_threading(file_paths):
    """Test the multiple resume parser with threading"""
    print(f"\n{'='*50}")
    print("Testing Multiple Resume Parser (Threading)")
    print(f"{'='*50}")

    url = f"{BASE_URL}/resume-parser-multiple"

    files = []
    for file_path in file_paths:
        with open(file_path, "rb") as file:
            files.append(("files", (os.path.basename(file_path), file.read())))

    start_time = time.time()
    response = requests.post(url, files=files)
    end_time = time.time()

    print(f"Status Code: {response.status_code}")
    print(f"Total Processing Time: {end_time - start_time:.2f} seconds")

    if response.status_code == 200:
        result = response.json()
        print(f"Total Files: {result['total_files']}")
        print(f"Successful: {result['successful_files']}")
        print(f"Failed: {result['failed_files']}")
        print(f"Success Rate: {result['summary']['success_rate']}%")
        print(
            f"Average Time per File: {result['summary']['avg_time_per_file']:.2f} seconds"
        )

        # Show results for each file
        for file_result in result["results"]:
            status = "✓" if file_result["success"] else "✗"
            print(f"  {status} {file_result['filename']}")
            if not file_result["success"]:
                print(f"    Error: {file_result['error']}")
    else:
        print(f"Error: {response.text}")


def test_multiple_resume_parser_multiprocessing(file_paths):
    """Test the multiple resume parser with multiprocessing"""
    print(f"\n{'='*50}")
    print("Testing Multiple Resume Parser (Multiprocessing)")
    print(f"{'='*50}")

    url = f"{BASE_URL}/resume-parser-multiple-mp"

    files = []
    for file_path in file_paths:
        with open(file_path, "rb") as file:
            files.append(("files", (os.path.basename(file_path), file.read())))

    start_time = time.time()
    response = requests.post(url, files=files)
    end_time = time.time()

    print(f"Status Code: {response.status_code}")
    print(f"Total Processing Time: {end_time - start_time:.2f} seconds")

    if response.status_code == 200:
        result = response.json()
        print(f"Total Files: {result['total_files']}")
        print(f"Successful: {result['successful_files']}")
        print(f"Failed: {result['failed_files']}")
        print(f"Success Rate: {result['summary']['success_rate']}%")
        print(
            f"Average Time per File: {result['summary']['avg_time_per_file']:.2f} seconds"
        )
        print(f"Workers Used: {result['workers_used']}")
        print(f"Performance Info: {result['summary']['performance_boost']}")

        # Show results for each file
        for file_result in result["results"]:
            status = "✓" if file_result["success"] else "✗"
            print(f"  {status} {file_result['filename']}")
            if not file_result["success"]:
                print(f"    Error: {file_result['error']}")
    else:
        print(f"Error: {response.text}")


def get_parser_info():
    """Get information about available endpoints"""
    print(f"\n{'='*50}")
    print("Resume Parser Information")
    print(f"{'='*50}")

    url = f"{BASE_URL}/resume-parser-info"
    response = requests.get(url)

    if response.status_code == 200:
        info = response.json()
        print(f"CPU Cores Available: {info['system_info']['cpu_cores']}")
        print(f"Supported Formats: {', '.join(info['supported_formats'])}")

        print("\nAvailable Endpoints:")
        for name, details in info["available_endpoints"].items():
            print(f"  • {details['endpoint']} - {details['description']}")
            print(f"    Use case: {details['use_case']}")
            print(f"    Max files: {details['max_files']}")

        print("\nRecommendations:")
        for scenario, recommendation in info["recommendations"].items():
            print(f"  • {scenario}: {recommendation}")
    else:
        print(f"Error: {response.text}")


def main():
    """Main function to run tests"""
    print("Resume Parser API Test Script")
    print("Make sure your FastAPI server is running!")

    # Get parser information first
    try:
        get_parser_info()
    except requests.exceptions.ConnectionError:
        print("\nError: Cannot connect to the API server.")
        print("Please make sure your FastAPI server is running on the specified URL.")
        return

    # Example usage - you'll need to provide actual file paths
    test_files = [
        # Add paths to your test resume files here
        # "path/to/resume1.pdf",
        # "path/to/resume2.docx",
        # "path/to/resume3.txt",
    ]

    if not test_files:
        print("\n" + "=" * 50)
        print("To test the endpoints, please:")
        print("1. Add resume file paths to the 'test_files' list in this script")
        print("2. Make sure your FastAPI server is running")
        print("3. Update the BASE_URL if needed")
        print("=" * 50)
        return

    # Verify files exist
    existing_files = [f for f in test_files if os.path.exists(f)]
    if not existing_files:
        print("\nNo test files found. Please check the file paths.")
        return

    print(f"\nFound {len(existing_files)} test files")

    # Test single file processing
    if existing_files:
        test_single_resume_parser(existing_files[0])

    # Test multiple file processing with threading
    if len(existing_files) > 1:
        test_multiple_resume_parser_threading(existing_files)

        # Test multiple file processing with multiprocessing
        test_multiple_resume_parser_multiprocessing(existing_files)


if __name__ == "__main__":
    main()
