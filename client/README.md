# PDF Form Processing - Web UI Client

See the **[Client User Guide](CLIENT_README.md)** for complete documentation.

## Quick Start

```bash
# Start the API server (from project root)
python server.py
# API runs on http://localhost:5000

# Start the client UI (from project root)
python client/src/client_server.py
# UI runs on http://localhost:5001
```

## Available Interfaces

| Interface | Local URL | Description |
|-----------|-----------|-------------|
| Main Client | http://localhost:5001 | Form management, canonical questions, PDF filling |
| Rating UI | http://localhost:5000/rate | Rate question similarity pairs |
| Rating Stats | http://localhost:5000/rate/stats | View rating statistics with CIs |
