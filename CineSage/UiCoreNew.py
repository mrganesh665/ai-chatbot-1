from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel
from typing import List, Optional
import streamlit as st

# ----------------------------
# Load Environment Variables
# ----------------------------
load_dotenv()

# ----------------------------
# Model
# ----------------------------
model = ChatMistralAI(model="mistral-small-2603")

# ----------------------------
# Pydantic Model
# ----------------------------
class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str

# ----------------------------
# Output Parser
# ----------------------------
parser = PydanticOutputParser(pydantic_object=Movie)

# ----------------------------
# Prompt
# ----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Extract movie information from the given paragraph.

{format_instructions}
"""
        ),
        (
            "human",
            "{paragraph}"
        )
    ]
)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🎬 Movie Information Extractor")

paragraph = st.text_area(
    "Enter Movie Paragraph",
    height=250
)

if st.button("Extract Information"):

    final_prompt = prompt.invoke(
        {
            "paragraph": paragraph,
            "format_instructions": parser.get_format_instructions(),
        }
    )

    response = model.invoke(final_prompt)

    movie_data = parser.parse(response.content)

    st.subheader("Movie Information")

    st.write(f"**Title:** {movie_data.title}")
    st.write(f"**Release Year:** {movie_data.release_year}")
    st.write(f"**Genre:** {', '.join(movie_data.genre)}")
    st.write(f"**Director:** {movie_data.director}")
    st.write(f"**Rating:** {movie_data.rating}")

    st.write("**Cast:**")
    for actor in movie_data.cast:
        st.write(f"- {actor}")

    st.write("**Summary:**")
    st.write(movie_data.summary)