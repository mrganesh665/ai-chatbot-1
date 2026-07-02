from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Personality Chatbot",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#eef2ff,#ffffff);
}

.title{
    font-size:48px;
    font-weight:800;
    text-align:center;
    color:#1f2937;
}

.subtitle{
    text-align:center;
    color:#6b7280;
    margin-bottom:30px;
}

.block-container{
    padding-top:2rem;
    max-width:1000px;
}

[data-testid="stSidebar"]{
    background:#111827;
}

[data-testid="stSidebar"] *{
    color:white;
}

.stChatMessage{
    border-radius:18px;
}

.mode-card{
    background:white;
    padding:15px;
    border-radius:15px;
    box-shadow:0px 5px 15px rgba(0,0,0,.08);
    margin-bottom:15px;
}

.mode-title{
    font-size:20px;
    font-weight:bold;
}

hr{
    margin-top:10px;
    margin-bottom:10px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    "<div class='title'>🤖 AI Personality Chatbot</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='subtitle'>Choose a personality and start chatting!</div>",
    unsafe_allow_html=True,
)

# -----------------------------
# Model
# -----------------------------
model = ChatMistralAI(
    model="mistral-small-2603",
    temperature=0.9
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🎭 AI Personality")

choice = st.sidebar.radio(
    "Choose Mode",
    [
        "😡 Angry",
        "😂 Funny",
        "😢 Sad"
    ]
)

if choice == "😡 Angry":
    mode = "You are an angry AI agent. You respond aggressively and impatiently."

elif choice == "😂 Funny":
    mode = "You are a funny AI agent. You respond with humor and jokes."

else:
    mode = "You are a sad AI agent. You respond in a very sad way."

st.sidebar.markdown("---")
st.sidebar.markdown("### Current Personality")
st.sidebar.success(choice)

# -----------------------------
# Session State
# -----------------------------
if (
    "messages" not in st.session_state
    or st.session_state.get("mode") != mode
):
    st.session_state.mode = mode
    st.session_state.messages = [
        SystemMessage(content=mode)
    ]

# -----------------------------
# Display Chat
# -----------------------------
for msg in st.session_state.messages:

    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)

    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# -----------------------------
# Input
# -----------------------------
prompt = st.chat_input("Type your message...")

if prompt:

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append(
        HumanMessage(content=prompt)
    )

    response = model.invoke(st.session_state.messages)

    st.session_state.messages.append(
        AIMessage(content=response.content)
    )

    with st.chat_message("assistant"):
        st.markdown(response.content)