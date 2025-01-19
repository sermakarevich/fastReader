# FastReader

FastReader is an application designed to process and summarize documents efficiently. It uses language models to compress text and generate both short and extensive summaries.

## Installation

To set up the environment and install dependencies, follow these steps:

1. **Create the environment:**
    ```sh
    make env-create
    ```

2. **Update the environment:**
    ```sh
    make env-update
    ```

3. **Remove the environment:**
    ```sh
    make env-remove
    ```

4. **Export the environment:**
    ```sh
    make env-export
    ```

5. **Install Jupyter kernel:**
    ```sh
    make jupyter-kernel
    ```

6. **Format the code:**
    ```sh
    make format
    ```

## Ollama Installation

Ollama is required for running local language models. To install Ollama, follow the instructions on the [Ollama GitHub page](https://github.com/ollama/ollama).

## Usage

To run the application, use the following command:

```sh
python src/run.py --URL https://www.zenml.io/blog/production-llm-security-real-world-strategies-from-industry-leaders --document_type text
```
