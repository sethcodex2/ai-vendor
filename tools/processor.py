from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from logging import getLogger

from langchain_community.document_loaders import CSVLoader, Docx2txtLoader, PyPDFLoader
from langchain_community.document_loaders import TextLoader
from aiohttp.client_exceptions import ClientConnectorError

from langchain_community.document_loaders import AsyncHtmlLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import OpenAI

import streamlit as st




# Load HTML
logger = getLogger(__name__)

class Processors:

    def __init__(self, api_key):
        self.openai_key = api_key
    
    def _save_embeddings(self, docs, docs_name):
        ''' creates embedding docs and save in postgres datastore '''
        embeddings = OpenAIEmbeddings(openai_api_key = self.openai_key)

        db = Chroma.from_documents(
            embedding = embeddings,
            documents = docs,
            collection_name = f'memo',
            persist_directory="./chroma_db"
            )
        


    def _get_loader(self, name, path):

        if name == 'csv':
            return CSVLoader(path)
        elif name == 'docx':
            return Docx2txtLoader(path)
        elif name == 'pdf':
            return PyPDFLoader(path)
        elif name == 'txt':
            return TextLoader(path)


    def get_best_documentation(self, question):
        print(question)
        embeddings = OpenAIEmbeddings(openai_api_key = self.openai_key)
        store = Chroma(
            embedding_function = embeddings,
            collection_name = f'memo',
            persist_directory="./chroma_db"
            )

        
        retriever = store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5, 'k': 1})

        llm = OpenAI(temperature=0, openai_api_key = self.openai_key)
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )

        compressed_docs = compression_retriever.get_relevant_documents(question)
        return '.\n'.join([doc.page_content for doc in compressed_docs])


                
    def process(self, loader,  document):
        loader = self._get_loader(loader, document.name)
        text_splitter = TokenTextSplitter(chunk_size=250, chunk_overlap=0)
        try:
            docs = loader.load_and_split(text_splitter)
            self._save_embeddings(docs, document.name)
            return docs[0].metadata['source']
        except:
            logger.exception('Error')
            return False
        
