# OD Map - PDF Form Processing API

A Flask-based API for processing, mapping, and filling PDF forms using AI-powered metadata extraction.

## Features

- **PDF Ingestion**: Download and store PDF forms from URLs to AWS S3
- **AI Metadata Extraction**: Extract form questions and fields using Google Gemini API
- **Canonical Question Mapping**: Create unified question sets across multiple forms
- **PDF Form Filling**: Fill PDF forms programmatically with mapped answers
- **Web UI Client**: Browser-based interface for testing and interacting with the API

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Client UI     │────▶│   Flask API     │────▶│   PostgreSQL    │
│  (localhost)    │     │   (server.py)   │     │   Database      │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │   AWS    │ │  Gemini  │ │  PyPDF   │
              │    S3    │ │   API    │ │   Form   │
              └──────────┘ └──────────┘ └──────────┘
```

## Project Structure

```
├── server.py              # Main Flask API server
├── database.py            # SQLAlchemy models and DB operations
├── s3_storage.py          # AWS S3 storage operations
├── extract_metadata.py    # PDF metadata extraction with Gemini
├── canonical_questions.py # Canonical question generation
├── form_mapping.py        # Form-to-canonical mapping logic
├── pdf_writer.py          # PDF form filling
├── client_fx.py           # PDF download client
├── gemini_client.py       # Gemini API client
├── batch_import.py        # Batch PDF import utility
├── migrate_to_db.py       # Database migration script
├── migrate_pdfs_to_s3.py  # S3 migration script
├── Procfile               # Heroku deployment config
├── requirements.txt       # Python dependencies
├── runtime.txt            # Python version for Heroku
└── client/                # Web UI client
    ├── src/
    │   └── client_server.py   # Client Flask server
    ├── templates/
    │   └── index.html
    └── static/
        ├── css/style.css
        └── js/app.js
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/pdf` | Add a PDF by URL |
| GET | `/api/forms/list` | List all forms |
| GET | `/api/forms/stats` | Get form statistics |
| POST | `/api/forms/metadata` | Get form metadata with canonical mappings |
| POST | `/api/forms/fill` | Fill a PDF form with answers |
| POST | `/api/canonical/create` | Create new canonical question version |
| GET | `/api/canonical/versions` | List canonical versions |
| GET | `/api/canonical/runs` | List canonical creation runs |

## Setup

### Prerequisites

- Python 3.12+
- PostgreSQL database
- AWS S3 bucket (via Heroku Bucketeer or direct)
- Google Gemini API key

### Environment Variables

Create a `.env` file in the root directory:

```env
# Database
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# AWS S3 (Bucketeer)
BUCKETEER_AWS_ACCESS_KEY_ID=your_access_key
BUCKETEER_AWS_SECRET_ACCESS_KEY=your_secret_key
BUCKETEER_BUCKET_NAME=your_bucket_name
BUCKETEER_AWS_REGION=us-east-1

# Gemini API
GOOGLE_API_KEY=your_gemini_api_key
```

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running Locally

### API Server

```bash
python server.py
```

The API runs on `http://localhost:5000`

### Client UI

In a separate terminal:

```bash
python client/src/client_server.py
```

The client UI runs on `http://localhost:5001`

#### Client Options

```bash
# Specify different ports
python client/src/client_server.py --port 5001 --api-url http://localhost:5000

# Connect to Heroku API
python client/src/client_server.py --api-url https://your-app.herokuapp.com
```

The client UI includes a toggle to switch between local and Heroku API endpoints.

## Deployment (Heroku)

```bash
# Add Heroku remote
heroku git:remote -a your-app-name

# Push to Heroku
git push heroku master

# Set environment variables
heroku config:set GOOGLE_API_KEY=your_key
heroku config:set DATABASE_URL=your_db_url
# Bucketeer vars are set automatically by the add-on
```

## Usage Flow

1. **Add PDFs**: POST PDF URLs to `/api/pdf` with opportunity IDs
2. **Extract Metadata**: System extracts questions and PDF fields using Gemini
3. **Create Canonical**: POST to `/api/canonical/create` to generate unified question set
4. **Get Metadata**: POST to `/api/forms/metadata` to get form with canonical mappings
5. **Fill Form**: POST answers to `/api/forms/fill` to generate filled PDF

## Database Tables

All tables are prefixed with `od1_`:

- `od1_forms` - Form metadata
- `od1_questions` - Questions within forms
- `od1_question_mappings` - Question-to-canonical mappings
- `od1_pdf_fields` - PDF form fields
- `od1_field_mappings` - PDF field to question mappings
- `od1_canonical_versions` - Canonical question versions
- `od1_canonical_questions` - Canonical question definitions

## License

Private - All rights reserved.
