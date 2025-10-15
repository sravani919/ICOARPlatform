
# tabs/assistant/assistant_tab.py
import streamlit as st
from tabs.CommonRasa.rasaui import send_to_rasa, render_rasa_message

def assistant_tab():
    st.subheader("Assistant (Rasa)")

    if st.session_state.get("authentication_status"):
        # Example use case
        user_input = st.text_input("Enter a message for Rasa:", key="assistant_input")
        if user_input:
            for m in send_to_rasa(user_input):
                render_rasa_message(m)

        with st.expander("Quick test: show downloads"):
            topic_test = st.text_input("Topic (optional):", key="topic_test")
            if st.button("Show downloads"):
                msg = "show downloads" if not topic_test else f"show downloads for {topic_test}"
                for m in send_to_rasa(msg):
                    render_rasa_message(m)
    else:
        # If not logged in, show a gentle nudge
        st.info("Please log in to use the assistant and dataset tools.")


