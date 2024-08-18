# Weaviate Document Store with FastAPI and Custom Llama Model

This repository contains a setup for a Weaviate document store, a FastAPI application for querying the document store, and a custom Llama model for generating detailed answers. The setup includes Docker Compose for container orchestration.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Configuration Details](#configuration-details)
- [Running the Application](#running-the-application)
- [Stopping the Application](#stopping-the-application)
- [Ingesting Documents](#ingesting-documents)
- [Querying the Document Store](#querying-the-document-store)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project sets up a Weaviate document store with a FastAPI application for querying and generating answers using a custom Llama model. The setup includes Docker Compose for container orchestration and various Python scripts for document ingestion and querying.

## Prerequisites

Before you begin, ensure you have the following installed on your machine:

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Python 3.8+](https://www.python.org/downloads/)

## Setup Instructions

1. **Clone the repository**:
    ```sh
    git clone https://github.com/5ud21/Weaviate-Document-Store.git
    cd Weaviate-Document-Store/
    ```

2. **Start the Docker Compose setup**:
    ```sh
    docker-compose up -d
    ```

3. **Install Python dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Configuration Details

The `docker-compose.yml` file defines the configuration for the Weaviate service. Below are the key configuration options:

- **Version**: Specifies the version of Docker Compose syntax being used (`3.4`).
- **Services**: Defines a single service named `weaviate`.
  - **Command**: Specifies the command-line arguments for the Weaviate service:
    - `--host`: Sets the host to `0.0.0.0`.
    - `--port`: Sets the port to `8080`.
    - `--scheme`: Sets the scheme to `http`.
  - **Image**: Uses the `semitechnologies/weaviate:1.21.2` Docker image.
  - **Ports**: Maps port `8080` on the host to port `8080` in the container.
  - **Volumes**: Mounts a Docker volume named `weaviate_data` to `/var/lib/weaviate` in the container.
  - **Restart Policy**: Configured to restart on failure (`on-failure:0`).
  - **Environment Variables**:
    - `QUERY_DEFAULTS_LIMIT`: Sets the default query limit to `25`.
    - `AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED`: Enables anonymous access (`true`).
    - `PERSISTENCE_DATA_PATH`: Sets the data path to `/var/lib/weaviate`.
    - `DEFAULT_VECTORIZER_MODULE`: Sets the default vectorizer module to `none`.
    - `ENABLE_MODULES`: Enables various modules for vectorization and generation.
    - `CLUSTER_HOSTNAME`: Sets the cluster hostname to `node1`.

## Running the Application

To start the Weaviate service, run the following command:

```sh
docker-compose up -d
```
This command will start the service in detached mode.

## Stopping the Application

To stop the Weaviate service, run the following command:

```sh
docker-compose down
```
This command will stop and remove the containers defined in the `docker-compose.yml` file.

## Ingesting documents

To ingest documents into the Weaviate document store, run the `ingest.py` script.

```sh
python ingest.py
```
This script performs the following steps:

-> Converts PDF documents into text.<br>
-> Preprocesses the text documents.<br>
-> Writes the preprocessed documents into the Weaviate document store.<br>
-> Updates the document embeddings using a specified embedding model.

## Querying the Document Store

To query the document store using the FastAPI application, run the `app.py` script:

```sh
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

This will start the FastAPI application on http://localhost:8001. You can use the /get_answer endpoint to query the document store and generate answers using the custom Llama model.

## Troubleshooting

### Common Issues

- **Port Conflicts**: Ensure that port `8080` is not being used by another service on your host machine.
- **Docker Daemon**: Ensure that the Docker daemon is running.

### Logs

To view the logs of the Weaviate service, run:

```sh
docker-compose logs weaviate
```

## License

This project is licensed under the MIT License. See the LICENSE file for more details. 
