# OD Map Client User Guide

This guide covers all the web interfaces available for interacting with the OD Map system.

## Quick Start

```bash
# Start the API server
cd C:\pp2\od_map
python server.py
# API runs on http://localhost:5000

# (Optional) Start the standalone client UI
python client/src/client_server.py
# Client runs on http://localhost:5001
```

## Web Interfaces

### 1. Main Client UI

**URL**: http://localhost:5001 (standalone) or use the API directly

The main client provides a tabbed interface for managing forms and testing the API.

#### Forms Tab

- **Add PDF**: Enter a file path on the server and an opportunity ID to ingest a new PDF
- **Form Statistics**: View aggregate stats across all forms (question counts, mapping rates)
- **Forms List**: Browse all forms with filtering by canonical version

#### Canonical Tab

- **Creation Runs**: Monitor background jobs that create canonical question sets
- **Create New Canonical**: Start a new canonical creation run (processes all forms with AI)
- **Canonical Versions**: View all versions with their question counts and match rates

#### Fill Form Tab

Step-by-step workflow to fill a PDF form:

1. **Select Form**: Choose a form by opportunity ID (auto-fills checksum)
2. **Select Version**: Choose which canonical version to use for mappings
3. **Set Min Similarity**: Questions with similarity >= this value use canonical text (1-5)
4. **Get Metadata**: Load the form's questions with canonical overrides
5. **Fill Answers**: Enter answers for each question
6. **Submit**: Generate the filled PDF and download

#### Debug Panel (Sidebar)

Shows all API requests and responses in real-time. Useful for:
- Understanding the API structure
- Debugging integration issues
- Copying request/response data

---

### 2. Similarity Rating UI

**URLs**:
| Environment | Rating | Statistics |
|-------------|--------|------------|
| Local | http://localhost:5000/rate | http://localhost:5000/rate/stats |
| Heroku | https://blurg-aeaf37f6c018.herokuapp.com/rate | https://blurg-aeaf37f6c018.herokuapp.com/rate/stats |

The rating UI allows human raters to evaluate LLM-assigned similarity scores.

#### Rating Page (`/rate`)

**Purpose**: Collect human judgments on whether question pairs are semantically equivalent.

**How to use**:
1. You'll see 10 question pairs (configurable)
2. Each pair shows "Question A" and "Question B"
3. For each pair, select:
   - **Same**: Both questions ask for the same information
   - **Not Same**: The questions are asking for different things
4. Click **Submit & Get More** to save and load the next batch

**Important notes**:
- The LLM's similarity score is hidden to avoid biasing your judgment
- Question IDs and form identifiers are hidden
- The system samples evenly across all similarity scores

#### Statistics Page (`/rate/stats`)

**Purpose**: View aggregate results of human ratings with confidence intervals.

**What you'll see**:
- **Summary**: Total number of ratings submitted
- **Whisker Plot**: Visual representation of human agreement by LLM score
  - X-axis: LLM similarity score (1-5)
  - Y-axis: % of pairs rated as "Same"
  - Dark blue bar: 50% confidence interval
  - Light blue bar: 95% confidence interval
  - Line: Mean percentage
- **Data Table**: Exact statistics for each score level

**Interpreting results**:
- Higher LLM scores should have higher % Same (if LLM is accurate)
- Narrow CI = more confident estimate (more ratings)
- Wide CI = uncertain estimate (need more ratings)

---

## Configuration

### Client Server Options

```bash
python client/src/client_server.py [OPTIONS]

Options:
  --port PORT       Port to run on (default: 5001)
  --api-url URL     API server URL (default: http://localhost:5000)
  --debug           Enable debug mode
```

### Rating UI Configuration

In `server.py`, the `RATING_CONFIG` controls:

```python
RATING_CONFIG = {
    "questions_per_page": 10,      # Questions per rating batch
    "bootstrap_iterations": 1000,  # CI calculation iterations
}
```

---

## API Toggle

The main client UI includes an API toggle in the header to switch between:
- **Local**: http://localhost:5000
- **Heroku**: https://blurg-aeaf37f6c018.herokuapp.com

This allows testing against either environment without restarting the client.

---

## Troubleshooting

### "No canonical version available"
- Create a canonical version first via the Canonical tab
- Or POST to `/api/canonical/create`

### Rating page shows no questions
- Ensure forms are mapped to the canonical version
- Check that forms have questions with similarity scores

### API connection failed
- Verify the API server is running
- Check the API URL in the toggle matches the running server
- For Heroku, ensure the app is awake (first request may be slow)

### Filled PDF not rendering correctly
- Try opening in Adobe Reader instead of Chrome
- Some PDF viewers don't display flattened form fields properly
