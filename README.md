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
    (Note - You might face issues while installing some packages (ex. better_profanity) via poetry. In such cases you can comment out the corresponding package name in pyproject.toml file and install it manually using pip (ex. pip install better_profanity))

4. Install custom packages.

    ```pip install streamlit_authenticator```

5. Create configuration files for the project in the `.streamlit` directory.

    - `.streamlit/secrets.toml`

    ```toml
    [api_token]
    twitter = "<your-twitter-api-token>"
    reddit = "<your-reddit-api-token>"
    ```
    For the contents of the secrets.toml file, please contact Pranav (psilimk@g.clemson.edu) or Mo (mshujaa@g.clemson.edu)
    - `.streamlit/authenticator.yaml`

    ```yaml
    credentials:
        usernames:
            atychang:
              email: allen.ty.chang@gmail.com
              name: Allen Chang
              password: $2b$12$I.E9myP3MfihzOSxrpkf8.O5S4xJuQNtxg7gd6Mdlzq6hXUVLEGQ. # hashed value of "password"
    cookie:
        expiry_days: 30
        key: some_signature_key
        name: some_cookie_name
    preauthorized:
        emails:
            - allen.ty.chang@gmail.com
    ```

6. Run the app

    ```shell
    streamlit run Home.py
    ```

7. Create folder /model in ICOAR project directory. </br>
   Download the image classification model from - https://drive.google.com/drive/folders/1q0bxYKrcQgEgg0Ps2N_WxgTApQK997g9 </br>
   Copy the model in the above directory.


8. Create folder /data/images/image in ICOAR project directory.

9. Download Node.js - https://nodejs.org/en/download/
    Generate a build for the react component. </br>
    Open a new terminal and run the below commands
   ```shell
   cd header_tab
   npm install
   npm run build
    ```
   ```shell
   cd citations
   npm install
   npm run build
    ```
      ```shell
   cd corousel
   npm install
   npm run build
    ```
    ```shell
   cd header_tab2
   npm install
   npm run build
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
