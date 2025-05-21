from http.server import BaseHTTPRequestHandler
import json
import os
import sys
import subprocess
from urllib.parse import parse_qs

def handler(event, context):
    """
    Serverless function handler for Vercel Python functions
    """
    # Parse the request body
    try:
        body = json.loads(event.get('body', '{}'))
    except:
        # If not JSON, try to parse as form data
        body = parse_qs(event.get('body', ''))
        # Convert lists to single values
        body = {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in body.items()}
    
    # Process the request
    if 'url' in body:
        # For YouTube URL classification
        result = classify_accent(body['url'], is_youtube_url=True)
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'X-Requested-With, Content-Type'
            },
            'body': json.dumps(result)
        }
    else:
        return {
            'statusCode': 400,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({"error": "No URL provided"})
        }

def classify_accent(url, is_youtube_url=False):
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
