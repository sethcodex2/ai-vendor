from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
from tools.bot import Bot


st.title("Vendor AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        bot = Bot(st.session_state['services'], key = st.session_state['key'])
        response = bot.generate_chat(prompt, st.session_state['username'])
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
