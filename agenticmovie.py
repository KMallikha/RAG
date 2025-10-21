from langchain_groq import ChatGroq
from langchain_community.document_loaders import CSVLoader
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.agents import initialize_agent,AgentType
from langchain.tools import Tool
from dotenv import load_dotenv
import os

load_dotenv()

apikey=os.getenv("GROQ_API_KEY")

docs=CSVLoader(file_path="movie.csv")
docdata=docs.load()

"""r=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100,separators=["\n\n", "\n", ".", " ", ""])
rp=r.split_documents(docdata)"""
rp=docdata

emb=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#vec_emb=Chroma.from_documents(documents=rp,embedding=emb,collection_name="moviedata",persist_directory="./chromadb")
db = Chroma(persist_directory="./chromadb", embedding_function=emb,collection_name="moviedata")
"""re=db.similarity_search("what is ocean pollution",k=2)
print(re)"""

llm=ChatGroq(api_key=apikey,model="groq/compound",temperature=0)
retriever= db.as_retriever(search_kwargs={"k":len(rp),"score_threshold":0.7})
template="""you are a helpful assistant 
-if user greets reply with greetings.
-if user asks question and if and only if strictly related to provided context, only answer the question based on given documents using the tools.
-if it is not related to the context strictly tell information is not related,i don't know
Do not guess or describe anything not present in the given document.
never use external knowledge or prior memory or internet

question:{question}
documents:{context}
"""

prompt=ChatPromptTemplate.from_template(template)
chain=RetrievalQA.from_chain_type(llm=llm,retriever=retriever,chain_type="stuff",chain_type_kwargs={"prompt":prompt},return_source_documents=True)

tools=[Tool(
    name="movierag",
    func=chain.run,
    description="""Use this tool to answer movie-related questions.
You MUST ONLY use the CSV data returned by the tool.
Do not answer any questions using your own knowledge.
If no answer is found in the CSV, respond with 'No data available."""
)]

agent=initialize_agent(tools=tools,llm=llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True,handle_parsing_errors=True)

while True:
    ip=input("Query: ")
    if ip=="Bye" or ip=="bye":
        print("goodbye")
        break
    result=agent.run(ip)
    print(result)