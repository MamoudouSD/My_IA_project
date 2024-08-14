import oci
from langchain_community.llms import OCIGenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.memory.buffer import ConversationBufferMemory
#from langchain.chains import LLMChain
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
import os
from uuid import uuid4
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain

unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"Test - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = ""

endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

llm= OCIGenAI(
    model_id ="cohere.command",
    service_endpoint = endpoint,
    compartment_id = "",
    model_kwargs = {"max_tokens":200}
)

embedding= OCIGenAIEmbeddings(
    model_id ="cohere.embed-english-v3.0",
    service_endpoint = endpoint,
    compartment_id = ""
)

db = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
retv = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})


template = """You are chatbot having a conversation with a human.
Human: {human_input}
AI: """

prompt = PromptTemplate(input_variables=["human_input"], template=template)

#la cle est utilise pour que si il y'a plusieur utilisateur les historique ne seront melange
history = StreamlitChatMessageHistory(key="chat_messages")

memory = ConversationBufferMemory(llm=llm, memory_key="chat_history", return_messages=True, output_key='answer')

model = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retv, memory=memory, return_source_documents=True)
#chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)


st.title('Welcome to the Diani chatbot')
for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

if x := st.chat_input():
    st.chat_message("human").write(x)

    response = model.invoke({"question":x})
    st.chat_message("ai").write(f"{response['answer']} {response['source_documents']}")
