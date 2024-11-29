# Crop and Weed Detection  

This repository provides a machine learning pipeline for detecting and classifying crops and weeds in agricultural images. It includes data preparation, model training, and a Flask-based GUI for real-time image classification.

## 📂 Folder Structure

```plaintext
Crop_and_Weed_Detection/
├── app.py                  # Flask application to connect frontend and backend
├── pascal_voc_format.csv   # Annotations in Pascal VOC format
├── images/                 # Test images for project
├── models/
│   └── crop_weed_model.keras   # Pre-trained MobileNetV2 model
├── data/
│   ├── agri_data/              # Raw images and YOLO label data
├── requirements.txt        # List of Python dependencies
├── templates/              # HTML files for the frontend
├── static/                 # Static assets like CSS, JS, and images
├── utils/                  # Utility scripts for preprocessing and visualization
│   ├──Cropand weed mobile.ipynb # Project file
│   ├── agri_data/              # Raw images and YOLO label data
└── README.md               # Documentation
```


## 🔍 Overview  

- **Objective**: Detect and classify agricultural objects as either **crops** or **weeds**.
- **Key Features**:
  - YOLO-label-to-Pascal-VOC data conversion.
  - Training models like **MobileNetV2** and **ResNet50**.
  - Flask-based GUI for user-friendly image classification.
- **Technologies Used**: Python, TensorFlow/Keras, OpenCV, Flask.

## 🚀 Setup Instructions  

1. Clone the repository:  ```bash git clone https://github.com/your-username/Crop_and_Weed_Detection.git
   cd Crop_and_Weed_Detection```
2. Install dependencies: ```pip install -r requirements.txt```
3. Run the Flask app:``` python app.py```
4. Open your browser and navigate to http://127.0.0.1:5000.

## 🧪 Training Details

1. Models:

MobileNetV2: Lightweight CNN optimized for mobile devices.
ResNet50: Deeper network for better accuracy.
2. Evaluation Metrics:

Accuracy, Precision, F1 Score, ROC-AUC.
3. Confusion Matrix Example:

## 📊 Results
MobileNetV2 Test Accuracy: 88%
ResNet50 Test Accuracy: 91%

## 👨‍💻 Author

Tiwari Mihir

Final-year Computer Engineering student, IAR

LinkedIn(www.linkedin.com/in/mihir-tiwari-7534b724b) | GitHub(https://github.com/MihirKT)

## 📄 License

This project is licensed under the MIT License. Feel free to use, modify, and distribute it as per the license terms.

## 🤝 Contributing

We welcome contributions! Please fork the repository and create a pull request with your changes.


For any questions or issues, open an issue in the repository.

Enjoy using Crop and Weed Detection! 😊




