from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import os

load_dotenv()

apikey=os.getenv("GROQ_API_KEY")

docs=[
    "The iPhone 16 was released in September 2024 with A18 chip.",
    "Apple launched iPhone 15 in 2023 with major camera upgrades."
]

emb=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",encode_kwargs={"device":"cpu","batch_size":8,"normalize_embeddings":True})
e=emb.embed_documents([docs[0]])

em=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
em.max_seq_length=10
dt=em.encode(docs[0],normalize_embeddings=True,device="cpu")
print(dt[0])
print(len(dt))


#vec_emb=FAISS.from_texts(docs,embedding=emb)