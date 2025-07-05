import os  # 1. System & Environment variables

from flask import Flask, render_template, jsonify, request  # 2. Web framework

# 3. Custom utilities and configuration
from medical_chatbot.utils.helper import download_huggingface_embeddings
from medical_chatbot.utils.embedding_utils import load_existing_pinecone_index
from medical_chatbot.config import PINECONE_API_KEY, MISTRAL_API_KEY
from medical_chatbot.indexer.constants import INDEX_NAME, EMBEDDING_DIMENSION
from medical_chatbot.prompt import system_prompt

# 4. LangChain tools and components
from langchain_pinecone import PineconeVectorStore
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# -------------------------#
# 🔧 Application Setup
# -------------------------#
app = Flask(__name__)

# -------------------------#
# 📦 Model & Vector Store Init
# -------------------------#
embed_model = download_huggingface_embeddings()

docsearch = load_existing_pinecone_index(
    index_name=INDEX_NAME,
    embed_model=embed_model
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

llm = ChatMistralAI(
    model="mistral-tiny",
    temperature=0.4,
    max_tokens=500
)

# -------------------------#
# 🧠 Prompt & RAG Chain Setup
# -------------------------#
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)

rag_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=question_answer_chain
)

# -------------------------#
# 🔄 Inference Endpoint
# -------------------------#

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])

# -------------------------#
# 🚀 Entry Point
# -------------------------#
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
