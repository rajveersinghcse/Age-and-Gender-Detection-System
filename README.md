# Age and Gender Detection System

This project utilizes OpenCV and deep learning models to detect and classify the age and gender of faces in images or video streams. It provides a real-time demonstration of age and gender recognition capabilities using pre-trained models.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- argparse

## Setup

1. Clone the repository:

```bash
git clone https://github.com/rajveersinghcse/Age-and-Gender-Detection-System.git
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. The necessary model files (`*.pb`, `*.prototxt`, `*.caffemodel`) are already included in the `constants/` directory. You don't need to download them separately.

## Usage

1. Run the script to start the age and gender detection:

```bash
python age_gender_detection.py --input path/to/input_file.ext --device cpu/gpu
```

- `--input`: Path to an input image or video file. Skip this argument to capture frames from a camera.
- `--device`: Specify the device for inference (`cpu` or `gpu`).

2. The program will display the input stream with bounding boxes indicating detected faces along with their predicted age and gender.

## Acknowledgments

This project is based on the OpenCV library and utilizes pre-trained deep learning models for face detection, age estimation, and gender classification.

## License

This project is licensed under the [MIT License](LICENSE).

Feel free to customize any part of this README.md file to better suit your project's details or style!
