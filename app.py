import os
from flask import Flask, request, render_template, session, redirect, url_for
from tqdm import tqdm

from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

app = Flask(__name__)
app.secret_key = 'mega_secretka_dlya_dvuh_chatov'

DOCUMENTS_FOLDER = 'documents'
VECTOR_STORE_PATH = 'vector_store'
os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

# Инициализируем модель один раз для всего приложения
llm = ChatOllama(model="mistral:7b-instruct-q4_K_M")

print("Инициализация быстрой модели для эмбеддингов...")
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

vector_store = None
retrieval_chain = None

def initialize_rag_chain():
    """Инициализирует цепочку RAG для ответов по документам."""
    global retrieval_chain
    
    prompt_text = """Ты — ассистент, который отвечает на вопросы, используя ТОЛЬКО предоставленный ниже контекст. Если в контексте нет информации для ответа на вопрос, ты ОБЯЗАН ответить только фразой: 'В предоставленных документах нет информации по этому вопросу.' Не используй свои общие знания.

    Контекст:
    {context}

    Вопрос: {input}
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    retriever = vector_store.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    print("Цепочка RAG успешно инициализирована.")

def build_vector_store():
    """Создает или загружает векторную базу данных."""
    global vector_store
    
    if os.path.exists(VECTOR_STORE_PATH):
        print("Загрузка существующей векторной базы...")
        vector_store = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)
    else:
        print("Создание новой векторной базы из папки 'documents'...")
        documents = []
        
        file_list = os.listdir(DOCUMENTS_FOLDER)
        for filename in tqdm(file_list, desc="Чтение документов"):
            filepath = os.path.join(DOCUMENTS_FOLDER, filename)
            try:
                if filename.endswith('.pdf'):
                    loader = PyPDFLoader(filepath)
                    documents.extend(loader.load())
                elif filename.endswith('.docx') or filename.endswith('.doc'):
                    loader = Docx2txtLoader(filepath)
                    documents.extend(loader.load())
            except Exception as e:
                print(f"\nНе удалось прочитать файл {filename}: {e}")

        if not documents:
            print("Документы для индексации не найдены.")
            return

        print("Разбиение текста на фрагменты...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        print("Создание эмбеддингов и сохранение в базу...")
        vector_store = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            persist_directory=VECTOR_STORE_PATH
        )
        print("Векторная база успешно создана и сохранена.")

    if vector_store:
        initialize_rag_chain()

@app.route('/', methods=['GET', 'POST'])
def rag_chat():
    """Маршрут для RAG-чата по документам."""
    if 'rag_history' not in session:
        session['rag_history'] = []

    if request.method == 'POST':
        question = request.form['question']
        
        if not retrieval_chain:
             ai_response = "База документов пуста или не инициализирована. Пожалуйста, добавьте файлы и перезапустите приложение."
        else:
            response = retrieval_chain.invoke({"input": question})
            ai_response = response["answer"]

        session['rag_history'].append({'user': question, 'ai': ai_response})
        session.modified = True
        return redirect(url_for('rag_chat'))

    return render_template('index.html', 
                           history=session['rag_history'], 
                           title="Чат по документам (RAG)",
                           action=url_for('rag_chat'))

@app.route('/general', methods=['GET', 'POST'])
def general_chat():
    """Маршрут для общего чата с AI."""
    if 'general_history' not in session:
        session['general_history'] = []

    if request.method == 'POST':
        question = request.form['question']
        
        # Просто отправляем вопрос в модель
        response = llm.invoke(question)

        # Убедимся, что ответ - это строка
        ai_response = response if isinstance(response, str) else response.content

        session['general_history'].append({'user': question, 'ai': ai_response})
        session.modified = True
        return redirect(url_for('general_chat'))
    
    return render_template('index.html', 
                           history=session['general_history'], 
                           title="Общий чат с AI",
                           action=url_for('general_chat'))

if __name__ == '__main__':
    build_vector_store()
    app.run(host='0.0.0', port=5000, debug=True)
