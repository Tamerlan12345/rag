import os
import json
from flask import (Flask, request, render_template, session, url_for,
                   send_from_directory, jsonify, redirect)
from tqdm import tqdm
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

app = Flask(__name__)
app.secret_key = 'adaptive_rag_agent_v6'

# --- КОНФИГУРАЦИЯ ---
DOCUMENTS_FOLDER = 'documents'
VECTOR_STORE_SUMMARIES_PATH = 'vector_store_summaries'
VECTOR_STORE_FULL_PATH = 'vector_store_full'
SUMMARIES_CACHE_FILE = 'summaries_cache.json'
os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

# --- ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ ---
llm = ChatOllama(model="qwen2:1.5b", timeout=600)
print("Инициализация модели для эмбеддингов...")
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

summary_vector_store = None
full_vector_store = None

# --- ЛОГИКА ПОСТРОЕНИЯ ВЕКТОРНЫХ БАЗ ---
def generate_summary(doc_content, filename):
    summary_prompt = ChatPromptTemplate.from_template("""Создай краткое резюме (1-2 предложения) для документа, описывающее его основную суть.
Документ: {filename}
Содержимое: {content}
Резюме:""")
    summary_chain = summary_prompt | llm
    return summary_chain.invoke({"filename": filename, "content": doc_content}).content

def build_vector_stores():
    global summary_vector_store, full_vector_store
    if os.path.exists(VECTOR_STORE_SUMMARIES_PATH) and os.path.exists(VECTOR_STORE_FULL_PATH):
        print("Загрузка существующих векторных баз...")
        summary_vector_store = Chroma(persist_directory=VECTOR_STORE_SUMMARIES_PATH, embedding_function=embeddings)
        full_vector_store = Chroma(persist_directory=VECTOR_STORE_FULL_PATH, embedding_function=embeddings)
    else:
        print("Создание новых векторных баз...")
        summaries_docs, full_docs_splits = [], []
        summaries_cache = {}
        if os.path.exists(SUMMARIES_CACHE_FILE):
            with open(SUMMARIES_CACHE_FILE, 'r', encoding='utf-8') as f: summaries_cache = json.load(f)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        for filename in tqdm(os.listdir(DOCUMENTS_FOLDER), desc="Обработка документов"):
            filepath = os.path.join(DOCUMENTS_FOLDER, filename)
            try:
                if filename.endswith('.pdf'): loader = PyPDFLoader(filepath)
                elif filename.endswith('.docx'): loader = Docx2txtLoader(filepath)
                else: continue
                docs = loader.load()
                if not docs: continue
                if filename not in summaries_cache:
                    print(f"\nГенерация резюме для: {filename}...")
                    summaries_cache[filename] = generate_summary(docs[0].page_content[:2000], filename)
                summary_doc = Document(page_content=summaries_cache[filename], metadata={"source": filename})
                summaries_docs.append(summary_doc)
                splits = text_splitter.split_documents(docs)
                for split in splits: split.metadata["source"] = filename
                full_docs_splits.extend(splits)
            except Exception as e: print(f"\nНе удалось обработать файл {filename}: {e}")
        if not summaries_docs: print("Не найдено подходящих документов."); return
        with open(SUMMARIES_CACHE_FILE, 'w', encoding='utf-8') as f: json.dump(summaries_cache, f, ensure_ascii=False, indent=4)
        print("Создание векторной базы для резюме..."); summary_vector_store = Chroma.from_documents(documents=summaries_docs, embedding=embeddings, persist_directory=VECTOR_STORE_SUMMARIES_PATH)
        print("Создание векторной базы для полного текста..."); full_vector_store = Chroma.from_documents(documents=full_docs_splits, embedding=embeddings, persist_directory=VECTOR_STORE_FULL_PATH)
    print("Векторные базы успешно созданы и готовы к работе.")


# --- ЛОГИКА АДАПТИВНОГО RAG-АГЕНТА ---

@app.route('/')
def rag_chat_page():
    if 'rag_history' not in session:
        session.clear() # Полная очистка для нового пользователя
        session['rag_history'], session['rag_state'] = [], 'INITIAL'
    return render_template('index.html', history=session['rag_history'], title="Адаптивный Агент", api_url=url_for('ask_rag'))

def is_complex_query(query):
    """Определяет, является ли запрос сложным."""
    # Простой эвристический метод: по количеству слов и наличию вопросительных слов
    words = query.split()
    question_words = ['какой', 'почему', 'как', 'сравни', 'объясни', 'опиши']
    if len(words) > 5 or any(word in query.lower() for word in question_words):
        return True
    return False

@app.route('/ask_rag', methods=['POST'])
def ask_rag():
    user_input = request.json.get('question')
    session['rag_history'].append({'user': user_input, 'time': request.json.get('time')})
    state = session.get('rag_state', 'INITIAL')
    ai_response = "Произошла непредвиденная ошибка. Состояние сброшено."

    def reset_state():
        keys_to_pop = ['rag_state', 'rag_original_question', 'rag_found_docs', 'rag_selected_doc']
        for key in keys_to_pop: session.pop(key, None)
        session['rag_state'] = 'INITIAL'

    if not summary_vector_store or not full_vector_store:
        ai_response = "База данных документов не загружена."
    
    # --- НАЧАЛО ДИАЛОГА: ВЫБОР ПУТИ (ПРОСТОЙ ИЛИ СЛОЖНЫЙ) ---
    elif state == 'INITIAL':
        # Если вопрос сложный, запускаем планировщик
        if is_complex_query(user_input):
            # ... (логика планировщика для сложных вопросов, как раньше) ...
            # Для текущей задачи мы ее упростим, чтобы сосредоточиться на быстром пути
            ai_response = "Это выглядит как сложный вопрос. Давайте попробуем найти документы."
            # Сразу переходим к поиску документов
            state = 'FINDING_DOCS'
        else:
            # Для простого вопроса сразу ищем документы
            state = 'FINDING_DOCS'
        
        # Общий код для поиска документов
        if state == 'FINDING_DOCS':
            retriever = summary_vector_store.as_retriever(search_kwargs={"k": 3})
            found_docs = retriever.invoke(user_input)
            doc_filenames = sorted(list(set([doc.metadata['source'] for doc in found_docs])))
            if not doc_filenames:
                ai_response = "К сожалению, я не нашел подходящих документов по вашему запросу."
                reset_state()
            else:
                session.update({'rag_state': 'AWAITING_DOC_CONFIRMATION', 'rag_found_docs': doc_filenames, 'rag_original_question': user_input})
                response_lines = ["Я нашел несколько потенциально подходящих документов. В каком из них мне искать ответ?", ""]
                for i, name in enumerate(doc_filenames): response_lines.append(f"{i+1}. {name}")
                response_lines.append("\nПожалуйста, укажите номер или название документа.")
                ai_response = "\n".join(response_lines)

    # --- ПОЛЬЗОВАТЕЛЬ ВЫБРАЛ ДОКУМЕНТ ---
    elif state == 'AWAITING_DOC_CONFIRMATION':
        found_docs, selected_doc_name = session.get('rag_found_docs', []), None
        try:
            choice_index = int(user_input.strip()) - 1
            if 0 <= choice_index < len(found_docs): selected_doc_name = found_docs[choice_index]
        except (ValueError, IndexError):
            for name in found_docs:
                if user_input.lower().strip() in name.lower(): selected_doc_name = name; break
        
        if selected_doc_name:
            original_question = session['rag_original_question']
            retriever = full_vector_store.as_retriever(search_kwargs={'k': 4, 'filter': {'source': selected_doc_name}})
            final_chunks = retriever.invoke(original_question)
            
            if not final_chunks:
                 ai_response = f"В документе «{selected_doc_name}» не найдено информации по вашему запросу."
            else:
                final_answer_prompt = ChatPromptTemplate.from_template("""Ты — ассистент-аналитик. Дай точный ответ на вопрос пользователя, основываясь ИСКЛЮЧИТЕЛЬНО на предоставленном контексте.

**Правила:**
1. Ответ должен быть СТРОГО в рамках контекста.
2. Если ответа нет, напиши только: 'В документе "{doc_name}" нет информации по этому вопросу.'
3. Структурируй ответ: **Ответ:**, **Цитата из документа:**, **Источник:**.

**Контекст из документа "{doc_name}":**
{context}

**Вопрос пользователя:** {input}""")
                answer_chain = create_stuff_documents_chain(llm, final_answer_prompt)
                response = answer_chain.invoke({"input": original_question, "doc_name": selected_doc_name, "context": final_chunks})
                ai_response = response
                download_link = url_for('download_file', filename=selected_doc_name)
                ai_response += f'\n\n[Скачать документ]({download_link})'
            reset_state()
        else:
             ai_response = "Не удалось распознать ваш выбор. Пожалуйста, укажите номер или название из списка."

    session['rag_history'].append({'ai': ai_response, 'time': request.json.get('time')})
    session.modified = True
    return jsonify({'answer': ai_response})


# --- Маршруты для Общего чата и утилиты (без изменений) ---
@app.route('/general')
def general_chat_page():
    if 'general_history' not in session:
        session['general_history'] = []
    return render_template('index.html', history=session['general_history'], title="Общий AI Ассистент", api_url=url_for('ask_general'))

@app.route('/ask_general', methods=['POST'])
def ask_general():
    user_input = request.json.get('question')
    time = request.json.get('time')
    session.setdefault('general_history', []).append({'user': user_input, 'time': time})
    messages = [
        SystemMessage(content="Ты — полезный AI-ассистент. Всегда отвечай на русском языке."),
        HumanMessage(content=user_input),
    ]
    response_obj = llm.invoke(messages)
    ai_text_response = response_obj.content
    session['general_history'].append({'ai': ai_text_response, 'time': time})
    session.modified = True
    return jsonify({'answer': ai_text_response})

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(DOCUMENTS_FOLDER, filename, as_attachment=True)

@app.route('/reset')
def reset():
    referrer = request.referrer
    if referrer and 'general' in referrer:
        session.pop('general_history', None)
        return redirect(url_for('general_chat_page'))
    else:
        keys_to_pop = ['rag_history', 'rag_state', 'rag_original_question', 'rag_found_docs', 'rag_selected_doc']
        for key in keys_to_pop:
            session.pop(key, None)
        return redirect(url_for('rag_chat_page'))

if __name__ == '__main__':
    build_vector_stores()
    app.run(host='0.0.0', port=5000, debug=True)
