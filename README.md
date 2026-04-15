# DFU Ensemble Screening

A web-based application for diabetic foot ulcer (DFU) screening using ensemble machine learning models. This application analyzes uploaded skin images to classify them as normal or abnormal skin conditions.

## Features

- **Multi-Model Ensemble**: Uses three pre-trained CNN models (MobileNetV2, EfficientNetB0, DenseNet121) with genetic algorithm optimization for ensemble weights
- **Visual Feature Analysis**: Incorporates image processing features like redness, contrast, edge density, and brightness ratios
- **Keyword-Based Classification**: Enhances predictions using domain-specific keywords for abnormal and normal skin conditions
- **Web Interface**: Simple Flask-based web application for image upload and results visualization
- **Real-time Processing**: Processes images and provides classification results with confidence scores

## Technology Stack

- **Backend**: Python Flask
- **Machine Learning**: TensorFlow/Keras with pre-trained models
- **Image Processing**: NumPy, Keras preprocessing
- **Frontend**: HTML/CSS with responsive design
- **Optimization**: Genetic algorithm for ensemble weight selection

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd dfu_screening_imple
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python main.py
   ```

4. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Upload a skin image (PNG, JPG, JPEG, GIF, BMP, WEBP formats supported)
2. The application will process the image using multiple CNN models
3. View the ensemble classification results including:
   - Skin assessment (Normal/Abnormal)
   - Confidence probabilities
   - Individual model predictions
   - Visual feature analysis

## Model Details

The application uses three pre-trained models with different optimizers:

- **MobileNetV2** with Adam optimizer
- **EfficientNetB0** with SGD optimizer
- **DenseNet121** with RMSprop optimizer

Ensemble weights are optimized using a genetic algorithm to maximize classification performance.

## Project Structure

```
dfu_screening_imple/
├── main.py                 # Flask application and ML logic
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html         # Web interface template
└── uploaded_images/       # Directory for uploaded images
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.