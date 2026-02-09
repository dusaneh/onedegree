# PDF Form Processing - Web UI Client

A web interface for interacting with the PDF form processing API.

## Quick Start

1. **Start the API server** (in the main project):
   ```bash
   cd C:\pp2\od_map
   python src/server.py
   ```
   API runs on http://localhost:5000

2. **Start the UI client** (in this folder):
   ```bash
   cd C:\pp2\od_map\client
   python src/client_server.py
   ```
   UI runs on http://localhost:5001

3. **Open the UI**: http://localhost:5001

## Features

### Forms Tab
- Add PDFs by file path or upload
- View all forms in the system
- Preview source PDFs

### Canonical Tab
- View canonical question versions
- Create new canonical version from existing forms

### Fill Form Tab
1. Select a form and canonical version
2. Click "Get Metadata" to load questions
3. Fill in answers (using canonical IDs for mapped questions, form IDs for others)
4. Click "Fill Form" to generate the filled PDF
5. Preview and download the result

### Debug Panel
- Shows all API requests and responses
- Expandable JSON view
- Useful for developers integrating with the API

## Configuration

```bash
# Custom ports
python src/client_server.py --port 5001 --api-url http://localhost:5000

# Debug mode
python src/client_server.py --debug
```

## Folder Structure

```
client/
├── src/
│   └── client_server.py    # Flask server
├── static/
│   ├── css/
│   │   └── style.css       # Styles
│   └── js/
│       └── app.js          # Frontend JavaScript
├── templates/
│   └── index.html          # Main HTML page
└── data/
    └── uploads/            # Uploaded PDFs
```
