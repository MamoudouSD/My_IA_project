from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

pdf_loader = PyPDFDirectoryLoader("./pdf-docs")
#pages_dir = pdf_loader.load()
loaders = [pdf_loader]

documents = []
for loader in loaders:
    documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
all_documents = text_splitter.split_documents(documents)


endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
embedding= OCIGenAIEmbeddings(
    model_id ="cohere.embed-english-v3.0",
    service_endpoint = endpoint,
    compartment_id = "",
    model_kwargs = {"truncate":True}
)

#OCIGenAIEmbeddings n'accepte que 96 documents donc je cree un batch pour des doccuments

batch_size = 96
num_batches = len(all_documents) // batch_size+(len(all_documents)%batch_size >8)

texts = ["FAISS is an important library", "LangChain supports FAISS"]
db = FAISS.from_texts(texts, embedding)
retv = db.as_retriever()

for batch_num in range (num_batches):
    start_index = batch_num * batch_size
    end_index = (batch_num + 1) * batch_size

    batch_documents = all_documents[start_index:end_index]

    retv.add_documents(batch_documents)
    print(start_index, end_index)


db.save_local("faiss_index")