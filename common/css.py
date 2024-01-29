import streamlit as st


def top_margin_20():
    st.markdown(
        """
        <div class="margin-20" style="margin-top: 15px;position: relative;opacity: 0;">20</div>
    """,
        unsafe_allow_html=True,
    )
