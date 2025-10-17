from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

apikey=os.getenv("GROQ_API_KEY")

llm=ChatGroq(api_key=apikey,model="groq/compound-mini",temperature=1)
while True:
    ip=input("Query: ")
    if ip=="Bye":
        break
    result=llm.invoke(ip)
    print(result.content)