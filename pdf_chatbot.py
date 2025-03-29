import sys
import os
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama  # Update to langchain-ollama if needed
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

# Load PDF
loader = PyPDFLoader("/Users/devg/Downloads/The_Memo-Rachel_Dodes.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  # Better settings
all_splits = text_splitter.split_documents(data)

# Create vectorstore
with SuppressStdout():
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

# Updated prompt template
template = """Answer the question based only on the following context:
{context}

Question: {question}
Answer concisely in 1-3 sentences. If unsure, say "I don't know"."""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Initialize LLM with updated parameters
llm = OllamaLLM(
    model="llama2",
    callbacks=[StreamingStdOutCallbackHandler()],
    temperature=0.3,
    system="Answer strictly based on the PDF context."
)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),  # Retrieve more chunks
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    return_source_documents=True
)

# Interaction loop
while True:
    query = input("\nQuery: ")
    if query.lower() == "exit":
        break
    if not query.strip():
        continue

    try:
        result = qa_chain.invoke({"query": query})  # Use invoke() instead of __call__
        if "result" in result:
            print("\nAnswer:", result["result"])
        else:
            print("I don't know the answer.")
    except Exception as e:
        print(f"Error: {str(e)}")
