import os
from flask import Flask, request, render_template, session, redirect, url_for
from langchain_ollama import ChatOllama, OllamaEmbeddings 
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)
app.secret_key = 'your_very_secret_key_for_chat'

DOCUMENTS_FOLDER = 'documents'
VECTOR_STORE_PATH = 'vector_store'
os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

llm = ChatOllama(model="phi3")
embeddings = OllamaEmbeddings(model="phi3")

vector_store = None

def build_vector_store():
    """
    Сканирует папку 'documents', обрабатывает файлы и создает/загружает векторную базу.
    """
    global vector_store
    
    if os.path.exists(VECTOR_STORE_PATH):
        print("Загрузка существующей векторной базы...")
        vector_store = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)
        return

    print("Создание новой векторной базы...")
    documents = []
    for filename in os.listdir(DOCUMENTS_FOLDER):
        filepath = os.path.join(DOCUMENTS_FOLDER, filename)
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
        elif filename.endswith('.docx'):
            loader = Docx2txtLoader(filepath)
            documents.extend(loader.load())

    if not documents:
        print("Документы для индексации не найдены.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    vector_store = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=VECTOR_STORE_PATH
    )
    print("Векторная база успешно создана и сохранена.")

@app.route('/', methods=['GET', 'POST'])
def chat():
    if 'history' not in session:
        session['history'] = []

    if request.method == 'POST':
        question = request.form['question']
        
        if not vector_store:
            ai_response = "База документов пуста. Пожалуйста, добавьте файлы в папку 'documents' и перезапустите приложение."
        else:
            prompt_text = """Ты — ассистент, который отвечает на вопросы, используя ТОЛЬКО предоставленный ниже контекст.
            - Твоя задача — найти ответ в тексте.
            - Если ответ найден, чётко изложи его своими словами на основе текста.
            - Если в контексте нет информации для ответа на вопрос, ты ОБЯЗАН ответить только фразой: "В предоставленных документах нет информации по этому вопросу."
            - Не используй свои общие знания. Не придумывай ничего.

            Контекст:
            {context}

            Вопрос: {input}
            """
            prompt = ChatPromptTemplate.from_template(prompt_text)

            retriever = vector_store.as_retriever()
            document_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            response = retrieval_chain.invoke({"input": question})
            ai_response = response["answer"]

        session['history'].append({'user': question, 'ai': ai_response})
        session.modified = True
        return redirect(url_for('chat'))

    return render_template('index.html', history=session['history'])

if __name__ == '__main__':
    build_vector_store()
    app.run(host='0.0.0', port=5000, debug=True)
