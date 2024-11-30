# Primary Disease Detector

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lusob/PrimaryDiseaseDetector/blob/main/PrimaryDiseaseDetector.ipynb)

This repository provides a TensorFlow-based pipeline for cancer primary disease classification using gene expression data from TCGA and MET500 datasets.

## Features

- Downloads and preprocess large datasets and phenotypes.
- Supports training a new model or using an existing pretrained model.
- Evaluates performance on the MET500 dataset using accuracy, classification reports, and confusion matrices.
- Converts gene expression data into image-like inputs for convolutional neural networks (CNNs).
- Grad-CAM Implementation: Generates heatmaps to highlight the most important genes or regions of image that contributed to the model’s predictions.

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

## Dataset Preprocessing

The [TCGA](https://drive.google.com/file/d/1-6OA1Q0TqFeooVHmURcZ_F9YjRh9D2cK/view?usp=drive_link) and [MET500](https://drive.google.com/file/d/1nBzGFuq-ExWw0KC0dtagJqAOFjji8bQc/view?usp=drive_link) datasets provided in Google Drive have been preprocessed as follows:

1. **Sources:**
   - The original TCGA and MET500 datasets were downloaded from:
     - **TCGA Gene Expression Data**: [Download](https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/tcga_RSEM_gene_fpkm.gz)
     - **MET500 Gene Expression Data**: [Download](https://ucsc-public-main-xena-hub.s3.us-east-1.amazonaws.com/download/MET500%2FgeneExpression%2FM.mx.log2.txt.gz)

2. **Steps:**
   - The datasets were intersected to include only common genes.
   - Both datasets were filtered to retain the intersected genes.
   - The data was saved as `log2(FPKM + 0.001)` values.

3. **Details:**
   - See the "Preprocessing Data" section in the notebook (`PrimaryDiseaseDetector.ipynb`) for the code used to generate these files.

4. **Download:**
   - The preprocessed datasets were uploaded to Google Drive and can be downloaded dynamically during notebook execution if `RETRAIN_MODEL=True`.

## Usage

### 1. Clone the repository

1. **Clone the repository:**
    ```bash
    git clone https://github.com/lusob/PrimaryDiseaseDetector.git
    cd PrimaryDiseaseDetector
    ```

### 2. Install dependencies

To install the required dependencies, run:
    ```bash
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
- Interactive visualizations of predictions over heatmaps created with grad-cam technique.

## Contributing

Feel free to fork this repository and submit pull requests for improvements.

## License

This project is licensed under the MIT License. See `LICENSE` for details.