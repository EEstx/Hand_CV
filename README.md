# AR Hand Gesture Control

Real-time 3D particle system controlled by hand gestures using computer vision and OpenGL rendering.

## Overview

This application uses MediaPipe for hand tracking and OpenGL for rendering interactive 3D particle-based shapes. All rendering is GPU-accelerated with bloom post-processing effects.

## Features

- 15 parametric 3D shapes (Cube, Torus, Pyramid, DNA, Heart, Mobius Strip, Klein Bottle, Sierpinski Fractal, etc.)
- Real-time hand gesture recognition
- GPU-accelerated particle system (configurable count)
- Bloom post-processing shader
- Multithreaded camera processing
- Smooth interpolation and transitions
- 60 FPS target performance

## System Requirements

### Software
- Python 3.8 - 3.11 (3.11 recommended)
- Webcam with 720p+ resolution
- OpenGL 2.0+ compatible GPU

### Dependencies
```
opencv-python
mediapipe
pygame
PyOpenGL
numpy
numba (optional, for performance boost)
```

## Installation

```bash
# Clone or download repository
cd Hand_cv

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install opencv-python mediapipe pygame PyOpenGL numpy

# Optional: Install numba for 10-50x performance boost
pip install numba
```

## Usage

```bash
python ar_main.py
```

### Controls

**Gestures:**
- Fist → Open Palm (within 1 second): Switch to next shape
- Pinch (thumb + index finger): Rotate object by moving hand
- No hand in frame: Auto-rotation

**Keyboard:**
- ESC: Exit application

## Configuration

All parameters are defined as constants at the top of `ar_main.py`:

```python
# Particles
PARTICLE_COUNT = 1000           # Number of particles
PARTICLE_SIZE = 10.0            # Point size in pixels
PARTICLE_SPEED = 0.16           # Transition speed (0.0-1.0)

# Bloom Effect
BLOOM_BLUR_SIZE = 3.0           # Blur radius
BLOOM_MULTIPLIER = 1.2          # Brightness multiplier

# Camera
CAMERA_INDEX = 0                # Camera device index
CAMERA_WIDTH = 1280             # Resolution width
CAMERA_HEIGHT = 720             # Resolution height

# Rotation
ROTATION_AUTO_SPEED = 0.5       # Auto-rotation speed (degrees/frame)
ROTATION_SENSITIVITY = 300      # Manual rotation sensitivity

# Performance
TARGET_FPS = 60                 # Target frames per second
```

## Available Shapes

1. Cube - Basic cube primitive
2. Torus - Donut shape
3. Pyramid - Four-sided pyramid
4. Atom - Nucleus with electron orbits
5. DNA - Double helix structure
6. Heart - Parametric heart surface
7. Mobius - Mobius strip topology
8. Star - Five-pointed star
9. Octahedron - Eight-sided platonic solid
10. Hyperboloid - One-sheet hyperboloid
11. Seashell - Logarithmic spiral shell
12. Wave - Sinusoidal surface
13. Klein Bottle - 4D Klein bottle projection
14. Sierpinski - 3D Sierpinski tetrahedron fractal
15. Super Torus - Square-profile torus

## Troubleshooting

### Camera not opening
- Verify camera connection and permissions
- Change `CAMERA_INDEX` constant (try 1 or 2)
- Check if another application is using the camera

### Low FPS
- Reduce `PARTICLE_COUNT` (try 500-2000)
- Close other GPU-intensive applications
- Disable bloom by setting `BLOOM_MULTIPLIER = 1.0`

### Gestures not recognized
- Ensure good lighting conditions
- Keep hand 30-60 cm from camera
- Use contrasting background
- Adjust `HAND_DETECTION_CONFIDENCE` if needed

### Import errors
- Use Python 3.11 for best compatibility
- Reinstall PyOpenGL: `pip install --upgrade PyOpenGL`
- On Linux: `sudo apt install python3-opengl`

## Architecture

### Core Components

- **HandTrackingThread**: Dedicated thread for camera capture and MediaPipe processing
- **ParticleSystem**: GPU-accelerated particle renderer with shape generators
- **BloomEffect**: Post-processing shader for glow effect
- **HandLandmarksRenderer**: Hand skeleton visualization with interpolation
- **GestureRecognizer**: Static gesture detection algorithms
- **WebcamBackground**: Camera feed texture renderer

### Performance Optimizations

- Threading: Camera processing decoupled from rendering
- Numba JIT: Particle position updates compiled to native code
- OpenGL VBOs: Batch rendering with vertex/color arrays
- Shader-based bloom: GPU gaussian blur implementation

## Technical Details

- Language: Python 3.11
- Graphics API: OpenGL 2.1 (compatibility profile)
- Shader Version: GLSL 120
- Hand tracking: MediaPipe Hands (21 landmarks)
- Video processing: OpenCV 4.x
- Math library: NumPy


