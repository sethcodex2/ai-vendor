from . import get_template_db
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from logging import getLogger
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from langchain_community.document_loaders import CSVLoader, Docx2txtLoader, PyPDFLoader
from langchain_community.document_loaders import TextLoader
from aiohttp.client_exceptions import ClientConnectorError

from langchain_community.document_loaders import AsyncHtmlLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import   LLMChain
from langchain_openai import OpenAI
from langchain_community.chat_message_histories import ChatMessageHistory

from tools.processor import Processors
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder

from langchain.agents import create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from .agent import CustomSearchTool
from langchain.agents import AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import PostgresChatMessageHistory

logger = getLogger(__name__)
template_engine = get_template_db()


no_reply_template = template_engine['business']['no_reply']
_template = template_engine['business']['system']
QUESTION_TEMPLATE = template_engine['question']
VALIDITY_TEMPLATE = template_engine['business']['validator']


def get_message_history(session_id):
	return PostgresChatMessageHistory(
            connection_string= st.secrets.connection,
            session_id=session_id,
            )

class Bot:

    def __init__(self, services, key):
        self.services = services
        self.openai_api_key = key

    def check_validity(self, message, context):
        llm = ChatOpenAI(temperature = 0, model = 'gpt-4', openai_api_key = self.openai_api_key)
        
        ai_template = SystemMessagePromptTemplate.from_template(VALIDITY_TEMPLATE)
        system_message = ai_template.format(services = '\n'.join([f'{i}) {service}' for i, service in enumerate(self.services)]))
        human_template = HumanMessagePromptTemplate.from_template('{human_input}')
        chat_template = ChatPromptTemplate.from_messages([system_message,  human_template])
    
        chain = LLMChain(llm = llm, prompt = chat_template, verbose = False)
        return chain.run(message)

    def generate_chat(self, question, session_id):
        logger.debug('started the generate chat process')
    

        logger.debug(f'KEY: {self.openai_api_key}')
        llm = ChatOpenAI(temperature = 0.1, model = 'gpt-4', openai_api_key = self.openai_api_key)
        template = _template
        
        processor = Processors(st.session_state['key'])
        _documentation = processor.get_best_documentation(question)
                      

        valid = self.check_validity(question, _documentation)
        if valid == 'N':
            #save client question as questions without answers

            # send no response to client
            logger.debug('generate_chat > failed validity check. Returning i cant reply')
            ai_template = SystemMessagePromptTemplate.from_template(no_reply_template)
            system_message = ai_template.format(
                question = question
                )
            _documentation = 'None'

        else:
            logger.debug('gnerate_chat > passed validity check')
            ai_template = SystemMessagePromptTemplate.from_template(template)
            system_message = ai_template.format(
            services = '\n'.join([f'{i}) {service}' for i, service in enumerate(self.services)])
            )

        prompt = MessagesPlaceholder(variable_name="chat_history")
        scratch_pad = MessagesPlaceholder(variable_name='agent_scratchpad')
        human_template = HumanMessagePromptTemplate.from_template('{human_input}')
        context_template = SystemMessagePromptTemplate.from_template(QUESTION_TEMPLATE)
        system_template_2 = context_template.format(context = _documentation)

        chat_template = ChatPromptTemplate.from_messages([system_message, system_template_2, prompt, human_template, scratch_pad])
        

        tools = [CustomSearchTool(), ]
        agent = create_openai_functions_agent(llm = llm,  tools = tools, prompt = chat_template)
        executor = AgentExecutor.from_agent_and_tools(  agent=agent, tools=tools)
        
        memory = get_message_history(session_id)

        agent_with_chat_history = RunnableWithMessageHistory(
            executor,
            # This is needed because in most real world scenarios, a session id is needed
            # It isn't really used here because we are using a simple in memory ChatMessageHistory
            lambda session_id: memory,
            input_messages_key="human_input",
            history_messages_key="chat_history",
            verbose = False
        )

        message = agent_with_chat_history.invoke(
            {"human_input":  question},
            config={"configurable": {"session_id": session_id}},
        )
        
        return message['output']

