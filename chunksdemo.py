from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os

load_dotenv()

docs= "The iPhone 16 was released in September 2024 with A18 chip.\n Apple launched iPhone 15 in 2023 with major camera upgrades."

c=CharacterTextSplitter(chunk_size=60,chunk_overlap=5,separator=". ")
cp=c.split_text(docs)
print(cp)

print("\n -----")
r=RecursiveCharacterTextSplitter(chunk_size=50,chunk_overlap=10,separators=["\n\n", "\n", ".", " ", ""])
rp=r.split_text(docs)
print(rp)

print("\n -----documents---")
docs1=PyPDFLoader("comp-off.pdf")
docdata=docs1.load()

cd=CharacterTextSplitter(chunk_size=500,chunk_overlap=100,separator=". ")
cpd=cd.split_documents(docdata)
print(cpd[1].page_content)

print("\n -----")
rd=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100,separators=["\n\n", "\n", ".", " ", ""])
rpd=rd.split_documents(docdata)
print(rpd[1].page_content)

