import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="Conversational Chatbot", page_icon="🤖", layout="centered")

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Conversational Chatbot")
st.markdown("### Chat with AI! Powered by LangChain & Groq 🚀")
st.divider()

@st.cache_resource
def load_model():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.7,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

llm = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.write(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = llm.invoke(st.session_state.messages)
                st.write(response.content)
                st.session_state.messages.append(AIMessage(content=response.content))
            except Exception as e:
                st.error(f"Error: {str(e)}")

with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    **Conversational Chatbot**
    
    - 🦜 LangChain Framework
    - ⚡ Groq AI (Llama3)
    - 🧠 Remembers conversation!
    - 🎨 Streamlit Interface
    """)
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

st.divider()
st.caption("Made with ❤️ using LangChain & Streamlit | © 2025")