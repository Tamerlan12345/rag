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

app = Flask(__name__)
app.secret_key = 'agent_analyst_rag_final_version'

# --- КОНФИГУРАЦИЯ ---
DOCUMENTS_FOLDER = 'documents'
VECTOR_STORE_SUMMARIES_PATH = 'vector_store_summaries'
VECTOR_STORE_FULL_PATH = 'vector_store_full'
SUMMARIES_CACHE_FILE = 'summaries_cache.json'
os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

# --- ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ ---
llm = ChatOllama(model="qwen2:1.5b", timeout=600) # Увеличим таймаут для сложных задач
print("Инициализация модели для эмбеддингов...")
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

summary_vector_store = None
full_vector_store = None

# --- ЛОГИКА ПОСТРОЕНИЯ ВЕКТОРНЫХ БАЗ ---
def generate_summary(doc_content, filename):
    """Генерирует краткое резюме для документа с помощью ИИ."""
    summary_prompt = ChatPromptTemplate.from_template("""Твоя задача — создать очень краткое, но информативное резюме (1-2 предложения) для документа. Резюме должно описывать основную суть и назначение документа, чтобы по нему можно было легко понять, о чем идет речь.

**Документ:** {filename}
**Начало содержимого:**
{content}

**Краткое резюме (1-2 предложения):**""")
    summary_chain = summary_prompt | llm
    response = summary_chain.invoke({"filename": filename, "content": doc_content})
    return response.content

def build_vector_stores():
    """Создает векторные базы, используя ИИ для генерации резюме документов."""
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
            with open(SUMMARIES_CACHE_FILE, 'r', encoding='utf-8') as f:
                summaries_cache = json.load(f)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        file_list = os.listdir(DOCUMENTS_FOLDER)
        for filename in tqdm(file_list, desc="Обработка документов"):
            filepath = os.path.join(DOCUMENTS_FOLDER, filename)
            try:
                if filename.endswith('.pdf'): loader = PyPDFLoader(filepath)
                elif filename.endswith('.docx'): loader = Docx2txtLoader(filepath)
                else: continue
                docs = loader.load()
                if not docs: continue
                if filename not in summaries_cache:
                    first_page_content = docs[0].page_content
                    print(f"\nГенерация резюме для: {filename}...")
                    summary_text = generate_summary(first_page_content[:2000], filename) # Ограничиваем объем для скорости
                    summaries_cache[filename] = summary_text
                else:
                    summary_text = summaries_cache[filename]
                summary_doc = Document(page_content=summary_text, metadata={"source": filename})
                summaries_docs.append(summary_doc)
                splits = text_splitter.split_documents(docs)
                for split in splits: split.metadata["source"] = filename
                full_docs_splits.extend(splits)
            except Exception as e:
                print(f"\nНе удалось обработать файл {filename}: {e}")
        if not summaries_docs or not full_docs_splits:
            print("Не найдено подходящих документов для индексации."); return
        with open(SUMMARIES_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(summaries_cache, f, ensure_ascii=False, indent=4)
        print("Создание векторной базы для резюме..."); summary_vector_store = Chroma.from_documents(documents=summaries_docs, embedding=embeddings, persist_directory=VECTOR_STORE_SUMMARIES_PATH)
        print("Создание векторной базы для полного текста..."); full_vector_store = Chroma.from_documents(documents=full_docs_splits, embedding=embeddings, persist_directory=VECTOR_STORE_FULL_PATH)
    print("Векторные базы успешно созданы и готовы к работе.")


# --- ЛОГИКА RAG-АГЕНТА ---

@app.route('/')
def rag_chat_page():
    if 'rag_history' not in session:
        session['rag_history'], session['rag_state'] = [], 'INITIAL'
    return render_template('index.html', history=session['rag_history'], title="Агент-Аналитик", api_url=url_for('ask_rag'))

@app.route('/ask_rag', methods=['POST'])
def ask_rag():
    user_input = request.json.get('question')
    session['rag_history'].append({'user': user_input, 'time': request.json.get('time')})
    state = session.get('rag_state', 'INITIAL')
    ai_response = "Произошла непредвиденная ошибка. Состояние сброшено."

    def reset_state():
        keys_to_pop = ['rag_state', 'rag_original_question', 'rag_search_plan', 'rag_selected_doc', 'rag_report_context']
        for key in keys_to_pop: session.pop(key, None)
        session['rag_state'] = 'INITIAL'

    if not summary_vector_store or not full_vector_store:
        ai_response = "База данных документов не загружена."
    
    # ЭТАП 1: Декомпозиция и планирование
    elif state == 'INITIAL':
        plan_prompt = ChatPromptTemplate.from_template("""Проанализируй сложный запрос пользователя. Разбей его на несколько простых, конкретных под-вопросов, чтобы составить план поиска информации в документах. План должен состоять из 2-5 пронумерованных шагов.

**Запрос пользователя:** {question}

**План поиска:**""")
        plan_chain = plan_prompt | llm
        plan_obj = plan_chain.invoke({"question": user_input})
        search_plan = plan_obj.content
        session.update({
            'rag_state': 'AWAITING_PLAN_CONFIRMATION',
            'rag_original_question': user_input,
            'rag_search_plan': search_plan
        })
        ai_response = f"Это комплексный вопрос. Чтобы дать максимально точный ответ, я предлагаю следующий план поиска:\n\n{search_plan}\n\nПродолжаем по этому плану? (да/нет)"

    # ЭТАП 2: Поиск документов по плану
    elif state == 'AWAITING_PLAN_CONFIRMATION':
        if user_input.lower().strip() in ['да', 'yes', 'продолжаем']:
            search_plan = session['rag_search_plan']
            retriever = summary_vector_store.as_retriever(search_kwargs={"k": 1})
            # Ищем самый релевантный документ для всего плана
            found_docs = retriever.invoke(session['rag_original_question'] + "\n" + search_plan)
            if not found_docs:
                ai_response = "К сожалению, я не смог найти подходящий документ для выполнения этого плана."
                reset_state()
            else:
                selected_doc = found_docs[0].metadata['source']
                session.update({
                    'rag_state': 'GENERATING_REPORT',
                    'rag_selected_doc': selected_doc
                })
                # Сразу переходим к генерации отчета, пропуская лишний вопрос
                ai_response = f"План подтвержден. Начинаю анализ документа: **{selected_doc}**. Пожалуйста, подождите, это может занять некоторое время..."
                # Мы отправляем этот промежуточный ответ, а реальная работа начнется после
        else:
            ai_response = "Поиск отменен. Задайте новый вопрос."
            reset_state()

    # ЭТАП 3: Синтез отчета
    elif state == 'GENERATING_REPORT':
        plan = session['rag_search_plan']
        doc_name = session['rag_selected_doc']
        original_question = session['rag_original_question']
        
        report_parts = [f"На основании документа «{doc_name}», вот анализ по вашему запросу:"]
        
        # Для каждого пункта плана ищем свою информацию
        for i, step in enumerate(plan.split('\n')):
            if not step.strip(): continue
            step_text = step.split('. ', 1)[-1] # Убираем номер
            
            retriever = full_vector_store.as_retriever(search_kwargs={'k': 3, 'filter': {'source': doc_name}})
            context_chunks = retriever.invoke(step_text)
            
            if not context_chunks:
                report_parts.append(f"\n\n**{step}**\n* В документе не найдено информации по этому пункту.")
                continue

            report_prompt = ChatPromptTemplate.from_template("""Ты — аналитик. Твоя задача — точно ответить на один пункт плана, используя только предоставленный контекст. Сформируй ответ, подкрепив его дословной цитатой и ссылкой на источник. Если в контексте нет ответа, напиши "Информация не найдена".

**Контекст:**
{context}

**Пункт плана для ответа:** {step}

**Отчет по пункту (Ответ, Цитата, Источник):**""")
            
            report_chain = report_prompt | llm
            report_obj = report_chain.invoke({"context": context_chunks, "step": step_text})
            report_parts.append(f"\n\n**{step}**\n{report_obj.content}")

        download_link = url_for('download_file', filename=doc_name)
        report_parts.append(f'\n\n---\n[Скачать документ: {doc_name}]({download_link})')
        ai_response = "\n".join(report_parts)
        reset_state()
    
    session['rag_history'].append({'ai': ai_response, 'time': request.json.get('time')})
    session.modified = True
    # На этапе генерации отчета мы возвращаем 'in_progress', чтобы фронтенд мог показать специальное сообщение
    if state == 'AWAITING_PLAN_CONFIRMATION' and user_input.lower().strip() in ['да', 'yes', 'продолжаем']:
         return jsonify({'answer': ai_response, 'action': 'start_report'})
    return jsonify({'answer': ai_response})

# --- Маршруты для Общего чата и утилиты ---
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
    response_obj = llm.invoke(user_input)
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
        keys_to_pop = ['rag_history', 'rag_state', 'rag_original_question', 'rag_search_plan', 'rag_selected_doc']
        for key in keys_to_pop:
            session.pop(key, None)
        return redirect(url_for('rag_chat_page'))

if __name__ == '__main__':
    build_vector_stores()
    app.run(host='0.0.0', port=5000, debug=True)
