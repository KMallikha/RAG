from langchain_groq import ChatGroq
from langchain_community.document_loaders import CSVLoader
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

apikey=os.getenv("GROQ_API_KEY")

docs=CSVLoader("rag_documents.csv")
docdata=docs.load()

r=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100,separators=["\n\n", "\n", ".", " ", ""])
rp=r.split_documents(docdata)

emb=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#vec_emb=Chroma.from_documents(documents=rp,embedding=emb,collection_name="mydata",persist_directory="./chromadb")
db = Chroma(persist_directory="./chromadb", embedding_function=emb,collection_name="mydata")
"""re=db.similarity_search("what is ocean pollution",k=2)
print(re)"""

llm=ChatGroq(api_key=apikey,model="groq/compound-mini")
retriever= db.as_retriever()
template="""you are a helpful assistant 
-if user greets reply with greetings.
-if user asks question and if and only if strictly related to provided context, answer the question using the Documents. 
-if it is not related to the context tell information is not related,i don't know"""

prompt=ChatPromptTemplate.from_messages([("system",template),("human","question:{question} documents:{context}")])
chain=RetrievalQA.from_chain_type(llm=llm,retriever=retriever,chain_type="stuff",chain_type_kwargs={"prompt":prompt},return_source_documents=True)

while True:
    ip=input("Query: ")
    if ip=="Bye":
        break
    result=chain.invoke(ip)
    print(result["result"])