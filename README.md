#PDF Question Answering with LangChain, Hugging Face & ChromaDB

### Overview
This project implements an **AI-powered PDF Question Answering System** using **LangChain**, **Hugging Face Transformers**, and **ChromaDB**.  

It allows users to:
- Upload and process PDF documents  
- Chunk and embed their content  
- Store embeddings in a persistent vector database  
- Retrieve relevant sections for a query  
- Generate contextual answers using **Mistral 7B Instruct**  
- Maintain **chat history** across multiple turns  

---

## Features
✅ Extracts text and metadata from PDF files  
✅ Splits text into overlapping chunks for better semantic retrieval  
✅ Embeds text using Sentence Transformers  
✅ Stores and retrieves embeddings using **ChromaDB**  
✅ Integrates **Mistral 7B** LLM from Hugging Face for natural responses  
✅ Keeps **session-based memory** to maintain conversational flow  

---

## Architecture
PDF Document → Text Splitter → Embeddings → Chroma Vector Store → Retriever → LLM (Mistral 7B) → Response

### 1.Install Dependencies
!pip install -U langchain
!pip install chromadb
!pip install pypdf

### 2.Load Environment Variables (API KEYS)
from dotenv import load_dotenv
load_dotenv()
For Google Colab:
from google.colab import userdata
userdata.get('HUGGINGFACEHUB_API_TOKEN')

### 3.Step-by-Step Implementation
1.Load pdf:
from  langchain_community.document_loaders import PyPDFLoader
pdf_path=r'/content/TechNet-One-Pager-on-AI-and-Gen-AI.pdf'
pdf=PyPDFLoader(pdf_path)
pdf_data=pdf.load()
#pdf_data[0]should print the metadatd n content of pdf page 1


2.Split text into chunks:
from langchain_text_splitters import CharacterTextSplitter
text_splitter=CharacterTextSplitter(chunk_size=200,chunk_overlap=20,separator='\n')
chunks=text_splitter.split_documents(pdf_data)
print(len(chunks)) #to check if there are chunks

3.Generate Embeddings:
from langchain_huggingface import HuggingFaceEmbeddings
model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
hf=HuggingFaceEmbeddings(
    model_name=model_name,
)

4.Vector store:

5.Create Retriever:
retriever = docsearch.as_retriever()

6.Intialize Hugging face llm:
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

llm_HFE = HuggingFaceEndpoint(
    repo_id='mistralai/Mistral-7B-Instruct-v0.2',
    task='text-generation',
    huggingfacehub_api_token=userdata.get('HUGGINGFACEHUB_API_TOKEN'),
    temperature=0.2,
    repetition_penalty=1.5,
    top_p=0.2,
    top_k=1,
    max_new_tokens=40
)
llm = ChatHuggingFace(llm=llm_HFE)

7.Define RAG Chain:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template('''
You are a helpful assistant.
Here is the previous conversation (if any):
{history}

Answer the following question based on the following context:
{context}

Question: {question}
''')

chain = (
    {
        "history": RunnableLambda(lambda x: x.get("history", "No previous conversation.")),
        "context": RunnableLambda(lambda x: retriever.invoke(x["question"])),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

8.Run the Chain:
result = chain.invoke('What is Artificial Intelligence?')
print(result)

9.Add chat memory:
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

_sessionstore = {}

def getsession_history(session_id: str):
    if session_id not in _sessionstore:
        _sessionstore[session_id] = InMemoryChatMessageHistory()
    return _sessionstore[session_id]

History_chain = RunnableWithMessageHistory(
    chain,
    getsession_history,
    input_messages_key="question",
    history_messages_key="history"
)

response = History_chain.invoke(
    {"question": "What is Artificial Intelligence?"},
    config={"configurable": {"session_id": "user1"}}
)
print(response)

response = History_chain.invoke(
    {"question": "What is Generative Artificial Intelligence?"},
    config={"configurable": {"session_id": "user1"}}
)
print(response)


10. Inspect stored Messages:
session_id = 'user1'
for message in _sessionstore[session_id].messages:
    print(message)

### 4. Example Output:
Artificial Intelligence (AI) refers to the ability of machines to perform tasks that typically require human intelligence such as learning, reasoning, and problem-solving.

##License

This project is open-source and available under the MIT License.

Author
Akshata Vyas
