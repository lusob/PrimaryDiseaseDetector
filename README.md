# Primary Disease Detector

This repository provides a TensorFlow-based pipeline for cancer classification using gene expression data from TCGA and MET500 datasets.

## Features

- Downloads large datasets and phenotypes directly from Google Drive (if training is required).
- Supports training a new model or using an existing pretrained model.
- Evaluates performance on the MET500 dataset using accuracy, classification reports, and confusion matrices.
- Converts gene expression data into image-like inputs for convolutional neural networks (CNNs).

## Repository Structure

- `data/`: Directory for datasets (downloaded dynamically if `RETRAIN_MODEL=True`).
- `model/`: Directory for saving and loading the trained model (`model/PrimaryDiseaseDetectorModel.keras`).
- `PrimaryDiseaseDetector.ipynb`: Main Jupyter Notebook.
- `README.md`: Instructions for setup and usage.

## Prerequisites

- Python 3.8 or higher
- Required Python libraries (install with `pip install -r requirements.txt`):
  - numpy
  - pandas
  - tensorflow
  - matplotlib
  - seaborn
  - sklearn
  - tqdm
  - gdown

## Usage

### 1. Clone the repository

1. **Clone the repository:**
    ```bash
    git clone https://github.com/lusob/PrimaryDiseaseDetector.git
    cd PrimaryDiseaseDetector
    ```

### 2. Install dependencies

To install the required dependencies, run:    ```bash
    pip install -r requirements.txt
    ```

### 3. Configure and Run the Notebook

Open `PrimaryDiseaseDetector.ipynb` and configure the variable `RETRAIN_MODEL`:

- **`RETRAIN_MODEL=True`**:
  - Downloads datasets and phenotypes from Google Drive.
  - Preprocesses the data.
  - Trains a new model and saves it in `model/PrimaryDiseaseDetectorModel.keras`.

- **`RETRAIN_MODEL=False`**:
  - Loads a pretrained model from `model/PrimaryDiseaseDetectorModel.keras`.
  - Skips data downloading and preprocessing.
  - Directly evaluates the model on the MET500 dataset.

### 4. Dataset Sources

The following large files are downloaded dynamically from Google Drive if `RETRAIN_MODEL=True`:

- **TCGA Gene Expression Data**: [Download](https://drive.google.com/file/d/1-6OA1Q0TqFeooVHmURcZ_F9YjRh9D2cK/view?usp=drive_link)
- **MET500 Gene Expression Data**: [Download](https://drive.google.com/file/d/1nBzGFuq-ExWw0KC0dtagJqAOFjji8bQc/view?usp=drive_link)
- **TCGA Phenotype Data**: [Download](https://drive.google.com/file/d/1wNXgjZMQUDqNosG_q8qZNIIq0za-ghF0/view?usp=drive_link)
- **MET500 Metadata**: [Download](https://drive.google.com/file/d/1-7yVlLwIo2aD_eojIysUllnRXb3j-b7e/view?usp=drive_link)

### 5. Results

The notebook generates:

- Model training and validation metrics.
- Accuracy, classification reports, and confusion matrices for the MET500 dataset.
- Visualizations of results.

## Contributing

Feel free to fork this repository and submit pull requests for improvements.

## License

This project is licensed under the MIT License. See `LICENSE` for details.