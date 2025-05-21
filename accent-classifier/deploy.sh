#!/bin/bash
# Simple deployment script for the accent classifier

echo "Preparing for Vercel deployment..."

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "Vercel CLI not found. Installing..."
    npm install -g vercel
fi

# Make sure all dependencies are installed
echo "Installing Node.js dependencies..."
npm install

echo "Installing Python dependencies..."
pip install -r api/requirements.txt

# Deploy to Vercel
echo "Deploying to Vercel..."
vercel --confirm

echo "Deployment initiated! Check the Vercel dashboard for status."
echo "To deploy to production, run: vercel --prod"
