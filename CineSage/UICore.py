from dotenv import load_dotenv
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

# -------------------------
# Load Environment Variables
# -------------------------
load_dotenv()

# -------------------------
# Model
# -------------------------
model = ChatMistralAI(
    model="mistral-small-2603"
)

# -------------------------
# Prompt
# -------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert Information Extraction Assistant.

Your task is to carefully analyze the given paragraph and extract all useful information.

Rules:
- Extract only explicitly mentioned information.
- Do not hallucinate.
- If information is missing, write "Not Mentioned".
- Organize the response with headings and bullet points.

Extract:
- Title / Name
- Category
- Genre
- Director
- Writer(s)
- Cast
- Plot
- Main Theme
- Ratings
- Important Facts
- Keywords
- Named Entities
- Quick Summary
- One-Line Summary
            """
        ),
        (
            "human",
            """
Extract useful information from the following paragraph:

{paragraph}
            """
        )
    ]
)

# -------------------------
# Streamlit Page
# -------------------------
st.set_page_config(
    page_title="Information Extractor",
    page_icon="📄",
    layout="centered"
)

# -------------------------
# CSS
# -------------------------
st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

h1 {
    text-align: center;
}

.result-box {
    background: #ffffff;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# Header
# -------------------------
st.title("📄 Information Extraction AI")
st.write("Paste a paragraph below to extract useful information.")

# -------------------------
# Input
# -------------------------
paragraph = st.text_area(
    "Paragraph",
    height=250,
    placeholder="Paste your paragraph here..."
)

# -------------------------
# Button
# -------------------------
if st.button("Extract Information", use_container_width=True):

    if paragraph.strip():

        final_prompt = prompt.invoke(
            {"paragraph": paragraph}
        )

        response = model.invoke(final_prompt)

        st.markdown("## 📋 Extracted Information")

        st.markdown(
            f"""
<div class="result-box">

{response.content}

</div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.warning("Please enter a paragraph.")