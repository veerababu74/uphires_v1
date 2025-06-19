# Multiple Resume Parser API

This API now supports processing multiple resume files simultaneously using threading or multiprocessing for improved performance.

## Available Endpoints

### 1. Single Resume Parser
**Endpoint:** `POST /resume-parser`
- **Use case:** Process a single resume file
- **Max files:** 1
- **Best for:** Testing, single file operations

```python
files = {'file': ('resume.pdf', open('resume.pdf', 'rb'))}
response = requests.post('http://localhost:8000/resume-parser', files=files)
```

### 2. Multiple Resume Parser (Threading)
**Endpoint:** `POST /resume-parser-multiple`
- **Use case:** I/O bound operations, moderate file counts
- **Max files:** 50
- **Best for:** File reading, API calls, network operations

```python
files = [
    ('files', ('resume1.pdf', open('resume1.pdf', 'rb'))),
    ('files', ('resume2.docx', open('resume2.docx', 'rb'))),
    ('files', ('resume3.txt', open('resume3.txt', 'rb')))
]
response = requests.post('http://localhost:8000/resume-parser-multiple', files=files)
```

### 3. Multiple Resume Parser (Multiprocessing)
**Endpoint:** `POST /resume-parser-multiple-mp`
- **Use case:** CPU-intensive operations, large file counts
- **Max files:** 100
- **Best for:** Heavy text processing, parsing operations

```python
files = [
    ('files', ('resume1.pdf', open('resume1.pdf', 'rb'))),
    ('files', ('resume2.docx', open('resume2.docx', 'rb'))),
    # ... up to 100 files
]
response = requests.post('http://localhost:8000/resume-parser-multiple-mp', files=files)
```

### 4. Parser Information
**Endpoint:** `GET /resume-parser-info`
- **Use case:** Get information about available endpoints and system capabilities

```python
response = requests.get('http://localhost:8000/resume-parser-info')
```

## Performance Comparison

| Method | Files | Use Case | Performance | Memory Usage |
|--------|-------|----------|-------------|--------------|
| Single | 1 | Testing, single operations | Baseline | Low |
| Threading | 1-50 | I/O bound tasks | 2-5x faster | Medium |
| Multiprocessing | 1-100 | CPU bound tasks | 3-10x faster | High |

## Response Format

All endpoints return a structured response:

```json
{
    "total_files": 5,
    "successful_files": 4,
    "failed_files": 1,
    "processing_time_seconds": 12.34,
    "results": [
        {
            "filename": "resume1.pdf",
            "success": true,
            "total_resume_text": "...",
            "resume_parser": {
                "name": "John Doe",
                "email": "john@example.com",
                "skills": ["Python", "FastAPI"],
                // ... other parsed data
            }
        },
        {
            "filename": "resume2.pdf",
            "success": false,
            "error": "Unsupported file type",
            "error_type": "UNSUPPORTED_FILE_TYPE"
        }
    ],
    "summary": {
        "success_rate": 80.0,
        "avg_time_per_file": 2.47,
        "performance_boost": "Used 4 CPU cores for parallel processing"
    }
}
```

## Error Handling

The API handles various error scenarios:

- **UNSUPPORTED_FILE_TYPE**: File extension not supported
- **NO_DATA_FOUND**: No resume data found in file
- **THREADING_ERROR**: Error in threading execution
- **MULTIPROCESSING_ERROR**: Error in multiprocessing execution

## Supported File Formats

- `.pdf` - PDF documents
- `.docx` - Microsoft Word documents
- `.txt` - Plain text files

## Performance Tips

### When to use Threading (`/resume-parser-multiple`):
- 1-50 files
- Files are stored on network drives
- API calls to external services
- I/O bound operations

### When to use Multiprocessing (`/resume-parser-multiple-mp`):
- 20+ files
- CPU-intensive text processing
- Large files that require significant parsing
- Maximum performance needed

### File Size Considerations:
- **Small files (<1MB)**: Threading is usually sufficient
- **Large files (>5MB)**: Multiprocessing provides better performance
- **Mixed sizes**: Use multiprocessing for consistency

## Example Usage

### Python Example

```python
import requests

# Upload multiple resumes with threading
files = []
resume_files = ['resume1.pdf', 'resume2.docx', 'resume3.txt']

for file_path in resume_files:
    with open(file_path, 'rb') as f:
        files.append(('files', (file_path, f.read())))

response = requests.post(
    'http://localhost:8000/resume-parser-multiple',
    files=files
)

result = response.json()
print(f"Processed {result['total_files']} files")
print(f"Success rate: {result['summary']['success_rate']}%")
```

### cURL Example

```bash
# Single file
curl -X POST "http://localhost:8000/resume-parser" \
     -F "file=@resume.pdf"

# Multiple files with threading
curl -X POST "http://localhost:8000/resume-parser-multiple" \
     -F "files=@resume1.pdf" \
     -F "files=@resume2.docx" \
     -F "files=@resume3.txt"

# Multiple files with multiprocessing
curl -X POST "http://localhost:8000/resume-parser-multiple-mp" \
     -F "files=@resume1.pdf" \
     -F "files=@resume2.docx" \
     -F "files=@resume3.txt"
```

## Security Considerations

- File size limits are enforced
- Temporary files are automatically cleaned up
- File type validation prevents malicious uploads
- Processing timeouts prevent resource exhaustion

## Monitoring and Logging

All endpoints include comprehensive logging:
- Processing times
- Success/failure rates
- Error details
- Performance metrics

Check the logs in the `logs/` directory for detailed information about processing results.

## Testing

Use the provided test script `test_multiple_resume_parser.py` to test all endpoints:

```bash
python test_multiple_resume_parser.py
```

Make sure to update the file paths in the script before running.
