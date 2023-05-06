# ICOAR

Integrative Cyberinfrastructure for Online Abuse Research (ICOAR)

## Installation

1. Make sure you have installed Python 3.8.1+ and [Poetry](https://python-poetry.org/) on your environment
2. Download the source code

    ```shell
    git clone https://github.com/CUSecLab/ICOAR.git
    cd ICOAR
    ```

3. Install the dependencies

    ```shell
    poetry install --only main
    ```

    This will create a virtual environment and install all the required dependencies.

    Now, you need to activate the virtual environment

    ```shell
    poetry shell
    ```

4. Create configuration files in `./streamlit`

    - `./streamlit/secrets.toml`

    ```toml
    [api_token]
    twitter = ""
    reddit = ""
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
