# Accent Classifier

A serverless application that classifies accents in audio/video content. This application allows users to either upload audio/video files or provide YouTube URLs for accent classification.

## Features

- Upload audio or video files for accent classification
- Provide YouTube URLs for accent classification
- Serverless architecture for easy deployment on Railway
- Modern, responsive UI

## Tech Stack

- **Frontend**: Next.js, React, TailwindCSS
- **Backend**: Next.js API Routes, Python
- **Audio Processing**: librosa, ffmpeg
- **Deployment**: Railway

## Getting Started

### Prerequisites

- Node.js (v14 or later)
- Python 3.8 or later
- ffmpeg (for audio processing)
- youtube-dl (for YouTube video processing)

### Installation

1. Clone the repository
2. Install JavaScript dependencies:
   ```
   npm install
   ```
3. Install Python dependencies:
   ```
   pip install -r api/requirements.txt
   ```

### Development

Run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser to see the application.

### Deployment

#### Deploying to Railway

This application is designed to be deployed on railway with both Next.js frontend and Python backend components.

1. **Install Railway CLI**:
   ```bash
   npm install -g railway
   ```

2. **Login to Railway**:
   ```bash
   railway login
   ```

3. **Deploy the Application**:
   ```bash
   # Navigate to the project directory
   cd accent-classifier
   
   # Deploy to Railway
   railway
   ```

4. **Deployment Configuration**:
   - When prompted, select "No" for linking to an existing project
   - Enter a project name (e.g., `accent-classifier`)
   - Select the root directory as the project directory
   - Railway will automatically detect Next.js and deploy accordingly

5. **Environment Variables** (if needed):
   You can set environment variables in the Railway dashboard or using the CLI:
   ```bash
   railway env add MY_VARIABLE
   ```

6. **Production Deployment**:
   After testing, deploy to production:
   ```bash
   railway --prod
   ```

#### Troubleshooting Railway Deployment

- **Python Dependencies**: If you encounter issues with Python dependencies, check the `api/requirements.txt` file and ensure all versions are compatible with Railway's Python runtime.

- **Execution Timeout**: For longer processing times, the `maxDuration` in `railway.json` is set to 30 seconds. If you need more time, consider optimizing your code or using external processing.

- **Memory Limits**: The application is configured with 1024MB of memory. If you need more, adjust the `memory` value in `railway.json`.

- **File Upload Size**: railway has a 4.5MB limit for API requests. For larger audio files, consider using client-side processing or external storage solutions.

This application is designed to be deployed on Railway. To deploy:

1. Install the Railway CLI:
   ```
   npm install -g railway
   ```

2. Deploy the application:
   ```
   railway
   ```

## How It Works

1. The user uploads a file or provides a YouTube URL
2. The application extracts audio from the file/URL
3. The audio is processed to extract relevant features
4. A machine learning model classifies the accent based on these features
5. The result is displayed to the user

## Supported Accents

The current version can classify the following accents:
- American
- British
- Australian
- Indian
- Spanish
- French
- German
- Russian
- Chinese
- Japanese

## License

MIT
