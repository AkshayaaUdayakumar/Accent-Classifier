#!/usr/bin/env python3
import sys
import json

def main():
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        is_youtube = input_path.startswith("http")
        
        # Provide a fallback response
        response = {
            "accent": "Unknown (fallback mode)",
            "confidence": 0.5,
            "message": "The accent classifier couldn't be fully loaded on this server.",
            "fallback": True
        }
        
        print(json.dumps(response))
        
if __name__ == "__main__":
    main()
