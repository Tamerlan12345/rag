# app.py
import os
from flask import Flask, request, render_template, session, redirect, url_for

# Импортируем необходимые компоненты из LangChain
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Инициализация Flask приложения ---
app = Flask(__name__)
app.secret_key = 'your_very_secret_key'  # Ключ для работы сессий

# --- Настройка путей ---
# Папка для загруженных файлов
UPLOAD_FOLDER = 'uploads'
# Папка для хранения векторной базы данных
VECTOR_STORE_FOLDER = 'vector_store'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_FOLDER, exist_ok=True)

# --- Главная страница ---
@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    # Если пользователь задал вопрос (отправил форму)
    if request.method == 'POST':
        # Проверяем, был ли ранее загружен файл и создана база
        if 'vector_store_path' in session:
            vector_store_path = session['vector_store_path']
            # Загружаем существующую векторную базу
            embeddings = OllamaEmbeddings(model="llama3")
            vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
            
            # Получаем вопрос пользователя из формы
            question = request.form['question']
            
            # --- Создание RAG-цепочки ---
            # 1. Определяем модель, которая будет генерировать ответ
            llm = ChatOllama(model="llama3")

            # 2. Создаем шаблон промпта. Он говорит модели, как использовать контекст.
            prompt = ChatPromptTemplate.from_template(
                """Ответь на вопрос пользователя, основываясь только на предоставленном контексте.
                
                Контекст:
                {context}
                
                Вопрос: {input}
                """
            )

            # 3. Создаем цепочку для "наполнения" промпта документами
            document_chain = create_stuff_documents_chain(llm, prompt)

            # 4. Создаем ретривер, который будет "доставать" релевантные документы из базы
            retriever = vector_store.as_retriever()
            
            # 5. Собираем всё в единую цепочку
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # 6. Запускаем цепочку и получаем ответ
            response = retrieval_chain.invoke({"input": question})
            answer = response["answer"]
        else:
            # Если файл еще не был загружен
            answer = "Пожалуйста, сначала загрузите PDF-документ."

    return render_template('index.html', answer=answer)

# --- Страница для загрузки файла ---
@app.route('/upload', methods=['POST'])
def upload_file():
    # Проверяем, есть ли файл в запросе
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
        
    if file and file.filename.endswith('.pdf'):
        # Сохраняем файл
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        # --- Обработка документа и создание векторной базы ---
        # 1. Загружаем PDF
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        
        # 2. Разбиваем текст на чанки
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # 3. Создаем эмбеддинги и сохраняем в ChromaDB
        # Указываем уникальный путь для этой сессии
        vector_store_path = os.path.join(VECTOR_STORE_FOLDER, file.filename + "_db")
        embeddings = OllamaEmbeddings(model="llama3")
        vector_store = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            persist_directory=vector_store_path
        )
        
        # Сохраняем путь к базе в сессии пользователя
        session['vector_store_path'] = vector_store_path
        
        return redirect(url_for('index'))

    return redirect(url_for('index'))

# --- Запуск приложения ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
