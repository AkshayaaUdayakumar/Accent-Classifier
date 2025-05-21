from http.server import BaseHTTPRequestHandler
import json
import os
import sys
import subprocess
from urllib.parse import parse_qs

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Get the size of data
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        # Parse the data
        try:
            data = json.loads(post_data.decode('utf-8'))
        except:
            # If not JSON, try to parse as form data
            data = parse_qs(post_data.decode('utf-8'))
            # Convert lists to single values
            data = {k: v[0] if len(v) == 1 else v for k, v in data.items()}
        
        # Set response headers
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type')
        self.end_headers()
        
        # Process the request based on path
        if self.path == '/api/classify':
            # Handle accent classification
            if 'url' in data:
                result = self.classify_accent(data['url'], is_youtube_url=True)
                self.wfile.write(json.dumps(result).encode('utf-8'))
            else:
                self.wfile.write(json.dumps({"error": "No URL provided"}).encode('utf-8'))
        else:
            self.wfile.write(json.dumps({"error": "Invalid endpoint"}).encode('utf-8'))
    
    def do_OPTIONS(self):
        # Handle preflight requests for CORS
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type')
        self.end_headers()
    
    def classify_accent(self, url, is_youtube_url=False):
        """
        Call the accent_classifier.py script to classify the accent
        """
        try:
            # Get the directory of this script
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Construct the path to accent_classifier.py
            classifier_path = os.path.join(current_dir, 'accent_classifier.py')
            
            # Make sure the script is executable
            os.chmod(classifier_path, 0o755)
            
            # Call the script
            cmd = [sys.executable, classifier_path, url]
            if is_youtube_url:
                print(f"Classifying YouTube URL: {url}")
            
            # Run the command and capture output
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Check for errors
            if result.returncode != 0:
                print(f"Error: {result.stderr}")
                return {"error": "Error classifying accent", "details": result.stderr}
            
            # Parse the JSON output
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {result.stdout}")
                return {"error": "Error parsing classification result", "output": result.stdout}
            
        except Exception as e:
            print(f"Exception: {str(e)}")
            return {"error": f"Error: {str(e)}"}
