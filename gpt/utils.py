import re

import streamlit as st


def escape_markdown(text: str) -> str:
    # Brought from https://github.com/python-telegram-bot/python-telegram-bot/blob/v20.2/telegram/helpers.py#L66
    escape_chars = r"\_*[]()~`>#+-=|{}.!$"
    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)


def key_directions():
    st.markdown(
        """**1: Input Your API Key**
            Use your API key and store it in the secrets.toml file like so:

            """
    )

    st.code(
        """
            [openai]
            api_key = (your key)
            """
    )
