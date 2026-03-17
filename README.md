# Face-Anonymizer
Real-time face detection and anonymization system using OpenCV DNN. This project captures webcam video, detects faces using a pre-trained deep learning model, and applies pixelation to blur faces dynamically for privacy protection.

# 1. Install dependencies

pip install opencv-python numpy

# 2. Download required model files

Download 'deploy.prototxt' and 'res10_300x300_ssd_iter_140000.caffemodel'

Put both files in the same folder as your 'main.py'

# 3. Run the script

python main.py

# 4. What happens

Webcam opens

Faces get detected

Faces are pixelated in real-time

Press 'q' to exit
