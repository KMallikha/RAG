from langchain_community.document_loaders import PyPDFLoader,TextLoader,CSVLoader, WebBaseLoader

"""document=PyPDFLoader('comp-off.pdf')
data=document.load()
print(data[1])

document1=PyPDFLoader("C:\\Users\\MallikhaKaparaveni\\Downloads\\water.pdf")
data1=document1.load()
print(data1[0])

document2=TextLoader(file_path="C:\\Users\\MallikhaKaparaveni\\OneDrive - Accellor\\Documents\\langchainfile.txt",encoding='utf-8')
data2=document2.load()
print(data2[0])

document3=CSVLoader("C:\\Users\\MallikhaKaparaveni\\Downloads\\day.csv")
data3=document3.load()
print(data3[5])

document4=WebBaseLoader(web_path="https://pub.towardsai.net/introduction-to-retrieval-augmented-generation-rag-using-langchain-and-lamaindex-bd0047628e2a")
data4=document4.load()
print(data4[0])
"""
