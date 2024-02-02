import streamlit as st
from replicate.client import Client

from common.css import top_margin_20


def input():
    label = "Enter your prompt below"
    prompt = st.text_area(
        label,
        value="",
        height=None,
        max_chars=None,
        key=None,
        help=None,
        on_change=None,
        args=None,
        kwargs=None,
        placeholder=None,
        disabled=False,
        label_visibility="visible",
    )

    return prompt


def generate_prompt():
    cols = st.columns(1)
    with cols[0]:
        prompt = input()
        top_margin_20()
        generate_prompt = st.button("Generate Prompt")
        system_prompt = """Can you optimize the prompt so that it gets better
        results? Please output the optimized prompt only"""

        if len(prompt) and generate_prompt:
            top_margin_20()
            with st.spinner("Generating Optimized Prompt..."):
                replicate = Client(api_token="r8_Zpdr150QkjHxb3yL6rLFDVLuwgfxfZD3XJuW4")
                output = replicate.run(
                    "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
                    input={
                        "debug": False,
                        "top_k": 50,
                        "top_p": 1,
                        "prompt": "Optimize the prompt - " + prompt,
                        "temperature": 0.5,
                        "system_prompt": system_prompt,
                        "max_new_tokens": 500,
                        "min_new_tokens": -1,
                    },
                )
                strOutput = ""
                for o in output:
                    strOutput += o
            st.write(strOutput)
        # print('Output - ', o)
