# Real-Time Scene Understanding with CLIP and OpenCV

## Overview
This repository implements a sophisticated real-time scene understanding system leveraging the **CLIP (Contrastive Language–Image Pretraining)** model from OpenAI, integrated with OpenCV for webcam-based video processing. The system generates descriptive captions for live video feeds by analyzing frames and matching them against a set of predefined candidate captions. This project showcases advanced computer vision and natural language processing techniques, optimized for low-latency performance.

Developed as part of an advanced EECS project at UC Berkeley, this work combines state-of-the-art multimodal AI with efficient video processing pipelines, making it suitable for applications in human-computer interaction, surveillance, augmented reality, and assistive technologies.

## Features
- **Real-Time Captioning**: Processes webcam video feeds to generate descriptive captions in real time.
- **CLIP Integration**: Utilizes the CLIP model to perform zero-shot image classification, enabling flexible and context-aware scene descriptions.
- **Optimized Performance**: Implements frame-skipping and lightweight preprocessing to ensure low-latency operation on consumer hardware.
- **Customizable Captions**: Supports a modular set of candidate captions, easily extensible for domain-specific applications.
- **User-Friendly Visualization**: Overlays captions directly on the video feed with OpenCV, providing an intuitive interface.

## Advantages Over Conventional Approaches
This system offers several advantages over traditional computer vision and scene description methods:

1. **Multimodal Understanding with CLIP**:
   - Unlike conventional object detection models (e.g., YOLO, SSD) that rely on predefined class labels, CLIP's ability to process both images and text enables zero-shot classification. This allows the system to generate nuanced captions without requiring retraining for new scenarios.
   - CLIP's pretraining on large-scale image-text pairs provides robust generalization across diverse scenes, outperforming traditional models in open-ended environments.

2. **Flexibility and Extensibility**:
   - The use of candidate captions allows easy customization for specific use cases (e.g., medical, industrial, or social settings) without modifying the core model.
   - Traditional systems often require extensive labeled datasets and retraining to adapt to new tasks, whereas this system can be reconfigured by updating the caption list.

3. **Real-Time Efficiency**:
   - By incorporating frame-skipping and optimized preprocessing, the system achieves real-time performance on standard hardware, unlike many deep learning-based captioning systems that require high-end GPUs or offline processing.
   - The lightweight integration of OpenCV ensures minimal computational overhead compared to frameworks like TensorFlow or PyTorch alone.

4. **Contextual Awareness**:
   - CLIP's joint image-text embedding space enables the system to capture high-level semantic relationships (e.g., actions, interactions, or scene compositions), surpassing conventional methods that focus solely on object detection or low-level features.
   - This results in more human-like descriptions, such as "Person performing an action" versus a simple object label like "Person."

5. **Robustness to Variability**:
   - Traditional computer vision systems struggle with variations in lighting, occlusion, or cluttered scenes. CLIP's transformer-based architecture and broad pretraining make it more resilient to such challenges, ensuring reliable captioning in real-world conditions.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/berkeley-eecs/realtime-clip-scene-understanding.git
   cd realtime-clip-scene-understanding
   ```

2. **Install Dependencies**:
   Ensure Python 3.8+ is installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

   Requirements include:
   - `opencv-python`
   - `numpy`
   - `pillow`
   - `torch`
   - `transformers`

3. **Hardware Requirements**:
   - A webcam (USB or built-in).
   - A CPU or GPU compatible with PyTorch (GPU recommended for faster processing).
   - At least 8GB RAM for smooth operation.

## Usage
1. **Run the Script**:
   ```bash
   python main.py
   ```
   - The webcam feed will open with real-time captions overlaid on the video.
   - Press `q` to exit the application.

2. **Customize Captions**:
   - Modify the `candidate_captions` list in `main.py` to include domain-specific descriptions.
   - Example: For a medical application, add captions like "Doctor examining patient" or "Medical equipment in use."

## Code Structure
- `main.py`: Core script implementing the real-time captioning pipeline.
- `requirements.txt`: List of required Python packages.
- `README.md`: Project documentation (this file).

## Example Output
When running the script, the webcam feed might display captions such as:
- "Person standing or sitting" (when a person is detected in a neutral pose).
- "Cluttered scene with multiple objects" (in a busy environment).
- "No person or object detected" (for empty scenes).

## Limitations
- **Computational Load**: While optimized, CLIP's inference can be resource-intensive on low-end hardware. Future work could explore model quantization or distillation.
- **Caption Granularity**: The system relies on predefined captions, which may not capture highly specific details. Fine-tuning CLIP or integrating a generative language model could address this.
- **Webcam Dependency**: The system requires a functional webcam and may need calibration for different devices.

## Future Work
- **Dynamic Caption Generation**: Replace static candidate captions with a generative model (e.g., GPT-4) for more detailed and context-specific descriptions.
- **Edge Deployment**: Optimize for edge devices like Raspberry Pi using lightweight models or TensorRT.
- **Multi-Modal Extensions**: Incorporate audio or depth sensors to enhance scene understanding.
- **Interactive Applications**: Integrate with AR/VR systems for real-time assistive or navigational aids.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for bug reports, feature requests, or optimizations. Ensure code follows PEP 8 standards and includes unit tests where applicable.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- Built using the **CLIP** model from OpenAI and **OpenCV**.
- Inspired by coursework in computer vision and machine learning at UC Berkeley EECS.
- Thanks to the open-source community for providing robust libraries and tools.