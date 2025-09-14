# Gigahack 2025 AI

This project provides a dockerized environment for training and running AI models.

## Prerequisites

- Docker
- Docker Compose

## Setup

1.  **Build the Docker image:**

    ```bash
    docker-compose build
    ```

## Usage

The application is managed through `docker-compose`.

### Training

To train a new model, you need to provide a dataset. The `data` directory is mounted into the container at `/app/data`. Place your training data in the `data` directory on your host machine.

To start the training process, run the following command, replacing the paths with your actual data and model paths:

```bash
docker-compose run --rm app python spacy-extract/train.py --dataset-path /app/data/your_dataset_folder --model-path /app/models/your_new_model
```

### Running Inference

To run inference with a trained model, you need to provide an input path and an output path. The `data` directory is mounted into the container, so you can use it for input and output.

To run inference, use the following command:

```bash
docker-compose run --rm app python evaluator.py --model-path /app/models/your_model --input-path /app/data/input_data --output-path /app/data/output_data
```

## Project Structure

- `spacy-extract/train.py`: Script for training a model.
- `evaluator.py`: Script for running inference.
- `Dockerfile`: Defines the Docker image for the application.
- `docker-compose.yaml`: Defines the services, volumes, and environment for the application.
- `data/`: Directory for your datasets and input/output data.
- `models/`: Directory for your trained models.

## Customization

You can run any python script inside the container using `docker-compose run --rm app python your_script.py`.
