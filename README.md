# Deepfake Video Player 🎬🔍

A real-time deepfake detection video player built with Python, PyQt5, and FastAPI. This application allows users to play videos while simultaneously detecting potential deepfake content using a pre-trained DenseNet121 model.

## ✨ Features

- **Real-time Video Playback**: Smooth video playback with standard controls (play/pause, seek)
- **Deepfake Detection**: AI-powered detection using a pre-trained DenseNet121 model
- **Live Analysis**: Real-time frame-by-frame analysis during video playback
- **User-friendly Interface**: Clean PyQt5-based GUI with intuitive controls
- **API-based Architecture**: FastAPI backend for scalable deepfake detection
- **Multiple Video Support**: Compatible with various video formats (MP4, AVI, etc.)

## 🏗️ Architecture

The project consists of two main components:

1. **FastAPI Backend** (`deepfake_api.py`): Handles deepfake detection requests
2. **PyQt5 Frontend** (`main.py`): Video player with real-time detection display

## 📋 Prerequisites

- Python 3.8+
- macOS/Linux/Windows
- Virtual environment support

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd Video_Player
```

### 2. Create and Activate Virtual Environment
```bash
# Create virtual environment
python -m venv venv310

# Activate virtual environment
# On macOS/Linux:
source venv310/bin/activate
# On Windows:
venv310\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: If `requirements.txt` is not available, install manually:
```bash
pip install fastapi uvicorn opencv-python PyQt5 keras tensorflow pillow requests numpy
```

## 🎯 Usage

### 1. Start the API Backend
```bash
# Terminal 1: Start the deepfake detection API
python deepfake_api.py
```
The API will be available at `http://127.0.0.1:8000`

### 2. Launch the Video Player
```bash
# Terminal 2: Start the video player application
python main.py
```

### 3. Using the Application
- **Load Video**: The player automatically loads `R4.mp4` by default
- **Playback Controls**: Use play/pause button and seek bar
- **Enable Detection**: Click "Enable Detection" to start real-time analysis
- **View Results**: Detection results appear as overlay text on the video

## 📁 Project Structure

```
Video_Player/
├── main.py                              # Main video player application
├── deepfake_api.py                      # FastAPI backend for detection
├── deepfake_detection_model(desnseNet121 Final).h5  # Pre-trained model
├── requirements.txt                      # Python dependencies
├── README.md                            # This file
├── venv310/                             # Virtual environment
├── sample_videos/                       # Sample videos for testing
│   ├── R1.mp4, R2.mp4, R3.mp4, R4.mp4  # Real videos
│   └── D1.mp4, D2.mp4, D3.mp4          # Deepfake videos
└── CMakeLists.txt                       # Build configuration
```

## 🔧 Configuration

### Model Configuration
- **Model File**: `deepfake_detection_model(desnseNet121 Final).h5`
- **Input Size**: 256x256 pixels
- **Threshold**: 0.5 (confidence > 0.5 = Deepfake, ≤ 0.5 = Real)

### API Configuration
- **Host**: 0.0.0.0
- **Port**: 8000
- **Endpoint**: `/predict/`

## 🧪 Testing

The project includes sample videos for testing:
- **Real Videos**: R1.mp4, R2.mp4, R3.mp4, R4.mp4
- **Deepfake Videos**: D1.mp4, D2.mp4, D3.mp4

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed in the virtual environment
2. **Model Loading**: Verify the model file exists and is accessible
3. **API Connection**: Check if the FastAPI server is running on port 8000
4. **Video Playback**: Ensure video files are in supported formats

### Debug Mode
Run with verbose logging:
```bash
python -u main.py
```

