from langchain_groq import ChatGroq
from langchain_community.document_loaders import CSVLoader
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import yaml
import os

load_dotenv()

apikey=os.getenv("GROQ_API_KEY")

"""docs=CSVLoader(file_path="moviecsv.csv",csv_args={"delimiter":","})
docdata=docs.load()

data={}
for doc in docdata:
    doc.page_content=(doc.page_content).replace("|",",")
    data=yaml.safe_load(doc.page_content)
    doc.metadata={
        "source":doc.metadata["source"],
        "row":doc.metadata["row"],
        **data
    }"""

emb=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

#vec_emb=Chroma.from_documents(documents=docdata,embedding=emb,collection_name="moviemetadata",persist_directory="./dbchroma")
db = Chroma(persist_directory="./dbchroma", embedding_function=emb,collection_name="moviemetadata")
#print(db.similarity_search("tell about toystory",k=2))
llm=ChatGroq(api_key=apikey,model="groq/compound")
retriever= db.as_retriever(search_kwargs={"k":20})
template="""you are a helpful assistant 
-if user greets reply with greetings.
-if user asks question and if and only if strictly related to provided context, answer the question using the Documents.
If the question requires comparison, analyze all given IMDb ratings and compute the result.
-if it is not related to the context strictly tell information is not related,i don't know
Do not guess or describe anything not present in the given document.
never use external knowledge or prior memory or internet"""

prompt=ChatPromptTemplate.from_messages([("system",template),("human","question:{question} documents:{context}")])
chain=RetrievalQA.from_chain_type(llm=llm,retriever=retriever,chain_type="stuff",chain_type_kwargs={"prompt":prompt},return_source_documents=True)

while True:
    ip=input("Query: ")
    if ip=="Bye" or ip=="bye":
        print("goodbye")
        break
    result=chain.invoke({"query":ip})
    print(result["result"])