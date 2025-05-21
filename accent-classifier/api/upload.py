import json
import os
import sys
import tempfile
import base64
import subprocess

def handler(event, context):
    """
    Serverless function handler for Vercel Python functions to handle file uploads
    """
    # Parse the request body
    try:
        body = json.loads(event.get('body', '{}'))
    except:
        return {
            'statusCode': 400,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({"error": "Invalid request format"})
        }
    
    # Check if file data is present
    if 'file' not in body or 'filename' not in body:
        return {
            'statusCode': 400,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({"error": "No file data provided"})
        }
    
    try:
        # Get file data (base64 encoded)
        file_data = body['file']
        filename = body['filename']
        
        # Create a temporary file to store the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            temp_path = temp_file.name
            
            # Decode and write the file content
            file_bytes = base64.b64decode(file_data.split(',')[1] if ',' in file_data else file_data)
            temp_file.write(file_bytes)
        
        # Process the file with accent_classifier.py
        result = classify_accent(temp_path)
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        # Return the result
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
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({"error": f"Error processing file: {str(e)}"})
        }

def classify_accent(audio_path):
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
