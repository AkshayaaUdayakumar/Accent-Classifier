from http.server import BaseHTTPRequestHandler
import json
import os
import sys
import tempfile
import subprocess
import cgi

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Parse form data with file
        content_type = self.headers.get('Content-Type', '')
        
        # Handle file uploads
        if 'multipart/form-data' in content_type:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST'}
            )
            
            # Check if file was uploaded
            if 'file' in form:
                file_item = form['file']
                
                # Create a temporary file to store the uploaded audio
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                    temp_path = temp_file.name
                    
                    # Write the file content
                    if file_item.file:
                        temp_file.write(file_item.file.read())
                
                # Process the file with accent_classifier.py
                result = self.classify_accent(temp_path)
                
                # Clean up the temporary file
                os.unlink(temp_path)
                
                # Send response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode('utf-8'))
                return
        
        # If we got here, something went wrong
        self.send_response(400)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({"error": "Invalid request"}).encode('utf-8'))
    
    def do_OPTIONS(self):
        # Handle preflight requests for CORS
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type')
        self.end_headers()
    
    def classify_accent(self, audio_path):
        """
        Call the accent_classifier.py script to classify the accent in an audio file
        """
        try:
            # Get the directory of this script
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Construct the path to accent_classifier.py
            classifier_path = os.path.join(current_dir, 'accent_classifier.py')
            
            # Make sure the script is executable
            os.chmod(classifier_path, 0o755)
            
            # Call the script
            cmd = [sys.executable, classifier_path, audio_path]
            
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
