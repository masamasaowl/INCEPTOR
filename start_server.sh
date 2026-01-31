#!/bin/bash

# INCEPTOR Voice Authentication - Startup Script
# This makes it easy to start everything!

echo "=================================================="
echo "🎙️  INCEPTOR - Voice Authentication System"
echo "=================================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed!"
    echo "   Please install Python 3.8 or higher"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"
echo ""

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed!"
    echo "   Please install pip3"
    exit 1
fi

echo "✅ pip3 found"
echo ""

# Install dependencies if needed
echo "📦 Checking Python dependencies..."
if ! python3 -c "import fastapi" &> /dev/null; then
    echo "   Installing dependencies (this might take a minute)..."
    pip3 install -r requirements.txt
    if [ $? -eq 0 ]; then
        echo "   ✅ Dependencies installed!"
    else
        echo "   ❌ Failed to install dependencies"
        echo "   Try running manually: pip3 install -r requirements.txt"
        exit 1
    fi
else
    echo "   ✅ Dependencies already installed"
fi

echo ""
echo "=================================================="
echo "🚀 Starting Voice Authentication Server..."
echo "=================================================="
echo ""
echo "📍 Server will run at: http://localhost:8000"
echo "📚 API docs at: http://localhost:8000/docs"
echo ""
echo "💡 TIP: Open a new terminal and run the frontend:"
echo "   cd frontend"
echo "   npm install  (first time only)"
echo "   npm run dev"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "=================================================="
echo ""

# Start the server
python3 server.py