"""
PDF Form Processing - Web UI Client

A web interface for interacting with the PDF form processing API.
Run this on a separate port (default: 5001) from the main API server (5000).

Usage:
    python client_server.py [--port 5001] [--api-url http://localhost:5000]
"""

import argparse
import base64
import json
import logging
import os
import requests
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("client_server")

# Flask app
app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

# Configuration
API_URLS = {
    "local": "http://localhost:5000",
    "heroku": "https://blurg-aeaf37f6c018.herokuapp.com"
}
API_BASE_URL = os.environ.get("API_URL", API_URLS["local"])
CLIENT_DATA_DIR = Path(__file__).parent.parent / "data"
UPLOAD_DIR = CLIENT_DATA_DIR / "uploads"

# Ensure directories exist
CLIENT_DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)


def api_request(method: str, endpoint: str, **kwargs) -> dict:
    """Make a request to the API server and return response with metadata."""
    url = f"{API_BASE_URL}{endpoint}"
    timeout = kwargs.pop("timeout", 300)

    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)

        try:
            response_data = response.json()
        except:
            response_data = {"raw": response.text}

        return {
            "success": response.ok,
            "status_code": response.status_code,
            "request": {
                "method": method,
                "url": url,
                "body": kwargs.get("json")
            },
            "response": response_data
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "status_code": 0,
            "request": {"method": method, "url": url, "body": kwargs.get("json")},
            "response": {"error": f"Cannot connect to API server at {API_BASE_URL}"}
        }
    except Exception as e:
        return {
            "success": False,
            "status_code": 0,
            "request": {"method": method, "url": url, "body": kwargs.get("json")},
            "response": {"error": str(e)}
        }


# =============================================================================
# Web UI Routes
# =============================================================================

@app.route("/")
def index():
    """Main UI page."""
    return render_template("index.html", api_url=API_BASE_URL)


@app.route("/api/config")
def get_config():
    """Get client configuration."""
    return jsonify({
        "api_url": API_BASE_URL,
        "api_urls": API_URLS,
        "upload_dir": str(UPLOAD_DIR)
    })


@app.route("/api/config/set-api-url", methods=["POST"])
def set_api_url():
    """Set the API URL dynamically."""
    global API_BASE_URL
    data = request.get_json()
    url_key = data.get("key")
    custom_url = data.get("url")

    if url_key and url_key in API_URLS:
        API_BASE_URL = API_URLS[url_key]
    elif custom_url:
        API_BASE_URL = custom_url
    else:
        return jsonify({"success": False, "error": "Invalid key or URL"}), 400

    return jsonify({"success": True, "api_url": API_BASE_URL})


# =============================================================================
# Proxy Routes to API Server
# =============================================================================

@app.route("/proxy/health")
def proxy_health():
    """Check API server health."""
    result = api_request("GET", "/api/health")
    return jsonify(result)


@app.route("/proxy/pdf", methods=["POST"])
def proxy_add_pdf():
    """Add a PDF to the system."""
    data = request.get_json()
    file_path = data.get("file_path")
    opp_id = data.get("opportunity_id")

    if not file_path or not opp_id:
        return jsonify({
            "success": False,
            "error": "Missing file_path or opportunity_id"
        }), 400

    result = api_request("POST", "/api/pdf", json={
        "file_path": file_path,
        "opportunity_id": opp_id
    })
    return jsonify(result)


@app.route("/proxy/pdf/upload", methods=["POST"])
def proxy_upload_pdf():
    """Upload a PDF file and add it to the system."""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400

    file = request.files["file"]
    opp_id = request.form.get("opportunity_id")

    if not file.filename or not opp_id:
        return jsonify({"success": False, "error": "Missing file or opportunity_id"}), 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = UPLOAD_DIR / filename
    file.save(filepath)

    # Add to API
    result = api_request("POST", "/api/pdf", json={
        "file_path": str(filepath),
        "opportunity_id": opp_id
    })

    return jsonify(result)


@app.route("/proxy/canonical/versions")
def proxy_get_versions():
    """Get canonical versions."""
    result = api_request("GET", "/api/canonical/versions")
    return jsonify(result)


@app.route("/proxy/canonical/create", methods=["POST"])
def proxy_create_canonical():
    """Create canonical questions."""
    result = api_request("POST", "/api/canonical/create", json={
        "webhook_url": f"{API_BASE_URL}/api/webhook/canonical"
    })
    return jsonify(result)


@app.route("/proxy/forms/metadata", methods=["POST"])
def proxy_get_metadata():
    """Get form metadata."""
    data = request.get_json()
    result = api_request("POST", "/api/forms/metadata", json=data)
    return jsonify(result)


@app.route("/proxy/forms/fill", methods=["POST"])
def proxy_fill_form():
    """Fill a form."""
    data = request.get_json()
    result = api_request("POST", "/api/forms/fill", json=data)
    return jsonify(result)


# =============================================================================
# PDF Preview & Download
# =============================================================================

@app.route("/proxy/pdf/preview")
def proxy_pdf_preview():
    """Get PDF as base64 for preview."""
    filepath = request.args.get("path")

    if not filepath or not os.path.exists(filepath):
        return jsonify({"success": False, "error": "PDF not found"}), 404

    try:
        with open(filepath, "rb") as f:
            pdf_data = base64.b64encode(f.read()).decode("utf-8")

        return jsonify({
            "success": True,
            "filename": os.path.basename(filepath),
            "data": pdf_data,
            "mime_type": "application/pdf"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/proxy/pdf/download")
def proxy_pdf_download():
    """Download a PDF file."""
    filepath = request.args.get("path")

    if not filepath or not os.path.exists(filepath):
        return jsonify({"success": False, "error": "PDF not found"}), 404

    return send_file(
        filepath,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=os.path.basename(filepath)
    )


@app.route("/proxy/forms/list")
def proxy_list_forms():
    """List all forms in the system via API."""
    version = request.args.get("version")
    endpoint = "/api/forms/list"
    if version:
        endpoint += f"?version={version}"

    result = api_request("GET", endpoint)
    return jsonify(result.get("response", {}))


@app.route("/proxy/forms/stats")
def proxy_form_stats():
    """Get form statistics via API."""
    version = request.args.get("version")
    endpoint = "/api/forms/stats"
    if version:
        endpoint += f"?version={version}"

    result = api_request("GET", endpoint)
    return jsonify(result.get("response", {}))


@app.route("/proxy/canonical/stats")
def proxy_canonical_stats():
    """Get canonical version statistics via API."""
    result = api_request("GET", "/api/canonical/stats")
    return jsonify(result.get("response", {}))


@app.route("/proxy/canonical/runs")
def proxy_canonical_runs():
    """Get all canonical creation runs."""
    result = api_request("GET", "/api/canonical/runs")
    return jsonify(result.get("response", {}))


@app.route("/proxy/canonical/run/<run_id>")
def proxy_canonical_run_status(run_id):
    """Get status of a specific canonical creation run."""
    result = api_request("GET", f"/api/canonical/run/{run_id}")
    return jsonify(result.get("response", {}))


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Form Processing Web UI")
    parser.add_argument("--port", type=int, default=5001, help="Port to run on (default: 5001)")
    parser.add_argument("--api-url", type=str, default="http://localhost:5000", help="API server URL")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()

    API_BASE_URL = args.api_url

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           PDF Form Processing - Web UI Client                ║
╠══════════════════════════════════════════════════════════════╣
║  UI Server:  http://localhost:{args.port}                         ║
║  API Server: {API_BASE_URL:<45} ║
╚══════════════════════════════════════════════════════════════╝
    """)

    app.run(host="0.0.0.0", port=args.port, debug=args.debug)
