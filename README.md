# Sentinel-2 pipeline for Mucilage Detection
The repository provides a pipeline for mucilage detection, including download of Sentinel-2 imagery, creation of the dataset and classification using an AI model.
Sentinel-2 satellite imagery are browsed and viewed from the Copernicus Data Space Ecosystem. You can search by coordinates, select various landscapes, and filter by cloud cover. This part was mainly inspired by [Sébastien Tétaud](https://github.com/sebastien-tetaud)'s repositories.
Sentinel-2 files are converted on the flight from SAFE to zarr.
Some data analytics are available in sentinel_analytics.ipynb.

## Installation

1. Clone the repository:

```bash
git clone git@github.com:germanasergi/mucilage-detection.git
cd mucilage-detection
```

2. Create and activate a conda environment:

```bash
conda create -n eopf python==3.11.7
conda activate eopf
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your credentials by creating a `.env` file in the root directory with the following content:

```bash
touch .env
```
then:

```
ACCESS_KEY_ID=username
SECRET_ACCESS_KEY=password
```



## Generate S2 dataset
```bash
python src/generate_dataset/generate_S2_dataset.py
```

## Download dataset
```bash
python src/generate_dataset/download_S2_dataset.py --config /mucilage-detection/data/adr_test/config_dataset_20250818_120134.yaml --l2a-csv /mucilage-detection/data/adr_test/output_l2a.csv
```

## Generate patches from images
```bash
python src/dataset/patches.py
```

## Classification

Perform classification of the patches using the model configuration defined in config.yaml
```bash
python src/main.py --patch_csv /path/to/patch/csv
```
