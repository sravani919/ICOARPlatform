import openai
import streamlit as st


def get_openai_key_from_user() -> str | None:
    st.caption("Enter your OpenAI API key to run this module (used only for this session).")
    k = st.text_input("OpenAI API Key", type="password")
    if k and k.strip():
        return k.strip()
    return None


def ask_gpt(text, prompt):
    key = get_openai_key_from_user()
    if not key:
        st.warning("Please enter your OpenAI API key to continue.")
        st.stop()

    openai.api_key = key

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{prompt}\n\nText: {text}"},
            ],
            max_tokens=50,
        )
        return response.choices[0].message["content"].strip().lower()
    except Exception as e:
        st.error(f"OpenAI error: {e}")
        return "error"


def process_prompts(text, prompts):
    for key in sorted(prompts.keys()):
        with st.spinner(f"Running {key}..."):
            response = ask_gpt(text, prompts[key])

        if response == "error":
            st.write("Final Decision: Analysis Incomplete")
            return "Analysis Incomplete"

        st.write(f"{key}: {response}")
        if "no" in response:
            st.write("Final Decision: Analysis Incomplete")
            return "Analysis Incomplete"

    st.write("Final Decision: Analysis Complete")
    return "Analysis Complete"


def in_context_learning():
    st.title("In-Context Learning with GPT")
    st.write("Enter the number of features and their descriptions.")

    # Key input at top (so user sees it first)
    _ = get_openai_key_from_user()

    num_features = st.number_input("Enter the number of features:", min_value=1, step=1)
    features = {}

    for i in range(1, int(num_features) + 1):
        feature = st.text_input(f"Enter the description for feature {i}:")
        if feature:
            features[f"Q{i}"] = feature.strip() + " Please answer only with 'yes' or 'no'."

    prompts = {}
    if features:
        prompts = {f"Q{i}": f"Does the text contain {features[f'Q{i}']}?" for i in range(1, len(features) + 1)}

    text_sample = st.text_area("Enter the text to analyze:")

    if st.button("Analyze"):
        if text_sample and features:
            final_decision = process_prompts(text_sample, prompts)
            st.write("Final Decision:", final_decision)
        else:
            st.warning("Please enter all required information.")