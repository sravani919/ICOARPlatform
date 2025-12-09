# visualization/__init__.py
from typing import Union

import pandas as pd
import streamlit as st


def add_graph_info(value_counts: Union[pd.Series, dict], data: pd.DataFrame) -> None:
    """
    Generic helper to show more information about the plotted labels.

    It works with ANY label column (sentiment, label, prediction, etc.)
    and does NOT assume a 'sentiment' column exists.
    """
    # Ensure value_counts is a Series
    if not isinstance(value_counts, pd.Series):
        value_counts = pd.Series(value_counts)

    # Try to infer which column was used for value_counts
    # If you did: value_counts = data[label_col].value_counts()
    # then value_counts.name should be label_col.
    label_col = value_counts.name if value_counts.name in data.columns else None

    st.markdown("### Dataset summary")

    st.write(f"Total rows in dataset: **{len(data)}**")
    st.write(f"Number of unique labels in this plot: **{len(value_counts)}**")

    st.markdown("#### Label distribution")
    st.dataframe(
        value_counts.rename("count").to_frame(),
        use_container_width=True,
    )

    # If we can't map back to a specific column, stop here
    if label_col is None:
        st.info(
            "Detailed examples are not available because the label column "
            "could not be inferred from the dataset."
        )
        return

    st.markdown("#### Explore a specific label")
    # Convert to string for the selectbox display
    label_values = value_counts.index.tolist()
    selected_label = st.selectbox(
        "Select a label to inspect examples",
        options=[str(v) for v in label_values],
    )

    # Filter rows for that label
    try:
        mask = data[label_col].astype(str) == selected_label
        subset = data[mask]
    except Exception:
        subset = data.iloc[0:0]  # empty

    st.write(f"Number of rows for this label: **{len(subset)}**")

    # Show example texts if available
    if "text" in subset.columns and not subset.empty:
        st.markdown("##### Example texts")
        for _, row in subset.head(5).iterrows():
            st.write(f"- {row['text']}")
