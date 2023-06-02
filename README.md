# ICOAR

Integrative Cyberinfrastructure for Online Abuse Research (ICOAR)

## Installation

1. Install Python 3.8.1+ and [Poetry](https://python-poetry.org/) if you haven't already.
2. Download the source code from the [ICOAR GitHub repository](https://github.com/CUSecLab/ICOAR).

    ```shell
    git clone https://github.com/CUSecLab/ICOAR.git
    cd ICOAR
    ```

3. Install the project dependencies

    ```shell
    poetry install --only main
    ```

    This command creates a virtual environment and installs all the required dependencies.
    To activate the virtual environment, run:

    ```shell
    poetry shell
    ```

4. Create configuration files for the project in the `.streamlit` directory.

    - `.streamlit/secrets.toml`

    ```toml
    [api_token]
    twitter = "<your-twitter-api-token>"
    reddit = "<your-reddit-api-token>"
    ```

5. Run the app

    ```shell
    streamlit run Home.py
    ```

## Contributing

1. Install development and testing dependencies

    ```shell
    poetry install
    ```

2. Install `pre-commit` hooks

    ```shell
    pre-commit install
    ```

3. Make your changes in a new branch and submit a pull request to the `main` branch.

test
