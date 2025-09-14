# Deployment and Usage Guide

This project provides a dockerized environment for training and running AI models. The setup is designed to be modular, allowing you to run different scripts for training, evaluation, and other tasks directly within a containerized environment.

## Prerequisites

Before you begin, ensure you have the following installed:

- Docker: [https://www.docker.com/get-started](https://www.docker.com/get-started)
- Docker Compose: [https://docs.docker.com/compose/install/](https://docs.docker.com/compose/install/)

## Setup

1.  **Build the Docker image:**

    This command builds the Docker image for the application based on the `Dockerfile`. It installs all the necessary dependencies defined in your `pyproject.toml` and `uv.lock` files.

    ```bash
    docker-compose build
    ```

## Usage

The application is managed through `docker-compose`. You can run any script within the containerized application.

### Training

To train a new model, you need to provide a dataset. The `data` directory in this project is mounted into the container at `/app/data`. This means you can place your training data in the `data` directory on your computer, and it will be accessible inside the container.

To start the training process, you will execute the `spacy-extract/train.py` script inside the container. Run the following command in your terminal, replacing the placeholder paths with your actual data and model paths:

```bash
docker-compose run --rm app python spacy-extract/train.py --dataset-path /app/data/your_dataset_folder --model-path /app/models/your_new_model
```

- `docker-compose run --rm app`: This command tells Docker Compose to run a one-off command in the `app` service. The `--rm` flag automatically removes the container after it exits.
- `python spacy-extract/train.py`: This is the command that will be executed inside the container.
- `--dataset-path`: Specifies the path to your training data inside the container.
- `--model-path`: Specifies the path where the trained model will be saved. Since the `models` directory is also mounted, the trained model will appear in the `models` directory on your computer.

### Running Inference

To run inference with a trained model, you will use the `evaluator.py` script. You need to provide paths for the model, input data, and where the output should be saved.

```bash
docker-compose run --rm app python evaluator.py --model-path /app/models/your_model --input-path /app/data/input_data --output-path /app/data/output_data
```

- `--model-path`: Path to the trained model you want to use.
- `--input-path`: Path to the input data for inference.
- `--output-path`: Path where the inference results will be saved.

## Project Structure

- `spacy-extract/train.py`: Script for training a model.
- `evaluator.py`: Script for running inference.
- `Dockerfile`: This file defines the Docker image for the application. It specifies the base Python image, sets up the working directory, installs dependencies, and copies your code into the image.
- `docker-compose.yaml`: This file is used to manage your Docker application. It defines the `app` service, configures the build process, and sets up volume mounts to share files between your computer and the container.
- `data/`: This directory is for your datasets and any other input or output data. It is mounted into the container.
- `models/`: This directory is for your trained models. It is also mounted into the container.

## Customization

The current setup is very flexible. You can run any Python script inside the container. For example, if you have another script called `my_script.py`, you can run it like this:

```bash
docker-compose run --rm app python my_script.py --your-argument value
```
