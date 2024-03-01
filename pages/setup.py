import streamlit as st
from  tools.processor import Processors
import tempfile
from datetime import datetime

datetime.now()


st.title('Setup Page')

username = st.text_input('Username')
key = st.text_input('Enter openai key')
services = st.text_input('Comma separated list of services')
file = st.file_uploader('Enter your knowledge data', type=['txt',])


if username and services and key and file is not None:

    doc = tempfile.NamedTemporaryFile(mode='wb+')
    doc.write(file.read())
    doc.seek(0)

    st.session_state['key'] = key
    st.session_state['username'] = username + str(datetime.now().timestamp()).replace('.','')
    st.session_state['services'] = services.strip().lower().split(',') 
    processor = Processors(key)
    response = processor.process('txt',doc)
    if response:
        st.success('Saved')
