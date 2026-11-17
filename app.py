from flask import Flask, request, jsonify, render_template
import os

print("🚀 Help desk...")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline 

app=Flask(__name__)

#Load Multiple PDFs
documents=[]
pdf_folder="uploads"
for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        pdf_path=os.path.join(pdf_folder,file)
        loader=PyPDFLoader(pdf_path)
        docs=loader.load()
        documents.extend(docs)

#text splitter
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
docs=text_splitter.split_documents(documents)

#Embeddings
embedding=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#vector Database
vectorstore=Chroma.from_documents(
    docs,
    embedding
)

#Retriever
retriever=vectorstore.as_retriever(
    search_kwargs={"k":4}
)
#load LLM
generator=pipeline(
    "text2text-generation",
     model="google/flan-t5-base",
    max_length=150
)

llm = HuggingFacePipeline(
    pipeline=generator
)

#QA chain
qa=RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True 
)

#student data
students = {
    "sam": "Sampath - CSE - 3rd Year - 9876543210",
    "rishi": "Rishi - ECE - 2nd Year - 9123456780",
    "ashu": "Ashwardh - ME - 4th Year - 9988776655"
}

#Home Route
@app.route("/")
def home():
    return render_template("index.html")

# Ask Route
@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({
            "answer": "Please send a question"
        }), 400

    question = data["question"].lower().strip()

# Greetings
    if question in ["hai", "hello", "hey"]:
        return jsonify({
            "answer": "Hello! 👋 Welcome to GenAI Academic Helpdesk."
        })

# Thank You
    if "thank" in question:
        return jsonify({
            "answer": "You're welcome 😊"
        })

# Bye
    if "bye" in question:
        return jsonify({
            "answer": "Goodbye 👋 Have a great day!"
        })
# Student Search
    words = question.split()

    for word in words:
        if word in students:
            return jsonify({
                "answer": students[word]
            })

# AI Query
    try:
        result = qa.invoke({
            "query": question
        })

        answer = result["result"]

        source = "Unknown"
        page = 0

        if result.get("source_documents"):
            doc = result["source_documents"][0]
            source = os.path.basename(
                doc.metadata.get("source", "Unknown")
            )
            page = doc.metadata.get("page", 0)

        return jsonify({
            "answer": answer,
            "source": source,
            "page": page
        })
    except Exception as e:
        return jsonify({
            "answer": f"Sorry, something went wrong: {str(e)}"
        }), 500

# Run Application
if __name__ == "__main__":
    app.run(debug=True)