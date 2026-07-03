from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

from langchain_mistralai import ChatMistralAI


model = ChatMistralAI( model = 'mistral-small-2603')

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

para = input("Give your paragraph : ")

final_prompt = prompt.invoke(
    {"paragraph": para}
)

res = model.invoke(final_prompt)

print(res.content)               



