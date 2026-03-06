import streamlit as st
import openai

def _get_openai_key() -> str | None:
    # 1) session key (user typed)
    k = st.session_state.get("OPENAI_API_KEY")
    if k:
        return k.strip()

    # 2) secrets (for your own dev)
    try:
        return st.secrets["openai"]["api_key"].strip()
    except Exception:
        return None

def _key_box():
    st.caption("Enter your OpenAI API key to run this module (used only for this session).")
    st.text_input("OpenAI API Key", type="password", key="OPENAI_API_KEY")

def generate_prompt():
    st.subheader("Prompt Optimization")
    _key_box()

    prompt = st.text_area("Enter your prompt below", value="", height=220)

    if st.button("Generate Optimized Prompt", use_container_width=True):
        api_key = _get_openai_key()
        if not api_key:
            st.error("Please enter your OpenAI API key.")
            return
        if not prompt.strip():
            st.error("Please enter a prompt.")
            return

        openai.api_key = api_key

        system_prompt = (
            "You are a prompt optimization assistant. "
            "Rewrite the user's prompt to be clearer, more specific, and more likely to get high-quality results. "
            "Return ONLY the optimized prompt. No extra text."
        )

        with st.spinner("Generating Optimized Prompt..."):
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=400,
                temperature=0.4,
            )

        optimized = resp.choices[0].message["content"].strip()
        st.text_area("Optimized Prompt", value=optimized, height=220)