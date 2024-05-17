import openai
import streamlit as st

openai.api_key = st.secrets["openai"]["openAI"]


def ask_gpt(text, prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}\n\nText: {text}"},
        ],
        max_tokens=50,
    )
    return response.choices[0].message["content"].strip().lower()


def process_prompts(text, prompts):
    responses = {}
    for key in sorted(prompts.keys()):
        response = ask_gpt(text, prompts[key])
        responses[key] = response
        st.write(f"{key}: {response}")
        if "no" in response:
            final_decision = "Analysis Incomplete"
            st.write(f"Final Decision: {final_decision}")
            return final_decision

    final_decision = "Analysis Complete"
    st.write(f"Final Decision: {final_decision}")
    return final_decision


def in_context_learning():
    st.title("In-Context Learning with GPT")
    st.write("Enter the number of features and their descriptions.'")

    num_features = st.number_input("Enter the number of features:", min_value=1, step=1)
    features = {}

    for i in range(1, num_features + 1):
        feature = st.text_input(f"Enter the description for feature {i}:")
        if feature:
            features[f"Q{i}"] = feature.strip() + " Please answer only with 'yes' or 'no'."

    if features:
        prompts = {f"Q{i}": f"Does the text contain {features[f'Q{i}']}?" for i in range(1, len(features) + 1)}

    text_sample = st.text_area("Enter the text to analyze:")

    if st.button("Analyze"):
        if text_sample and features:
            final_decision = process_prompts(text_sample, prompts)
            st.write("Final Decision:", final_decision)
        else:
            st.write("Please enter all required information.")
