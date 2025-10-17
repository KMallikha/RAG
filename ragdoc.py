from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

apikey=os.getenv("GROQ_API_KEY")

docs=PyPDFLoader("comp-off.pdf")
docdata=docs.load()

r=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100,separators=["\n\n", "\n", ".", " ", ""])
rp=r.split_documents(docdata)

emb=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vec_emb=FAISS.from_documents(documents=rp,embedding=emb)

llm=ChatGroq(api_key=apikey,model="groq/compound-mini",temperature=1)
retriever= vec_emb.as_retriever()
template="""you are a helpful assistant 
-if user greets reply with greetings.
-if user asks question if related to provided context answer the question using the Documents. 
-if it is not related to the context tell information is not related,i don't know"""

prompt=ChatPromptTemplate.from_messages([("system",template),("human","question:{question} documents:{context}")])
chain=RetrievalQA.from_chain_type(llm=llm,retriever=retriever,chain_type="stuff",chain_type_kwargs={"prompt":prompt},return_source_documents=True)

"""while True:
    ip=input("Query: ")
    if ip=="Bye":
        break
    result=chain.invoke(ip)
    print(result["result"])"""