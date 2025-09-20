import os
import json
from flask import Flask, request, render_template, session, url_for, send_from_directory, jsonify, redirect
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
app.secret_key = 'smartest_rag_agent_v5'

# --- КОНФИГУРАЦИЯ ---
DOCUMENTS_FOLDER = 'documents'
VECTOR_STORE_SUMMARIES_PATH = 'vector_store_summaries'
VECTOR_STORE_FULL_PATH = 'vector_store_full'
SUMMARIES_CACHE_FILE = 'summaries_cache.json'
os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

# --- ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ ---
llm = ChatOllama(model="qwen2:1.5b", timeout=300)
print("Инициализация модели для эмбеддингов...")
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

summary_vector_store = None
full_vector_store = None

# --- УЛУЧШЕНИЕ №1: СОЗДАНИЕ "УМНОГО" СЛОЯ МЕТАДАННЫХ ---
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
    
    # Проверяем, нужно ли пересоздавать базы
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

                # Создаем резюме или берем из кэша
                if filename not in summaries_cache:
                    first_page_content = docs[0].page_content
                    print(f"\nГенерация резюме для: {filename}...")
                    summary_text = generate_summary(first_page_content, filename)
                    summaries_cache[filename] = summary_text
                else:
                    print(f"\nИспользование кэшированного резюме для: {filename}")
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
        
        # Сохраняем кэш резюме
        with open(SUMMARIES_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(summaries_cache, f, ensure_ascii=False, indent=4)
        
        print("Создание векторной базы для резюме..."); summary_vector_store = Chroma.from_documents(documents=summaries_docs, embedding=embeddings, persist_directory=VECTOR_STORE_SUMMARIES_PATH)
        print("Создание векторной базы для полного текста..."); full_vector_store = Chroma.from_documents(documents=full_docs_splits, embedding=embeddings, persist_directory=VECTOR_STORE_FULL_PATH)
    
    print("Векторные базы успешно созданы и готовы к работе.")


# --- ЛОГИКА ЧАТА ---

@app.route('/')
def rag_chat_page():
    # ... (код без изменений) ...
    if 'rag_history' not in session:
        session['rag_history'], session['rag_state'] = [], 'INITIAL'
    return render_template('index.html', history=session['rag_history'], title="Интерактивный RAG Агент", api_url=url_for('ask_rag'))


@app.route('/ask_rag', methods=['POST'])
def ask_rag():
    user_input = request.json.get('question')
    # Убираем `.lower()` чтобы сохранить регистр для лучшего распознавания
    session['rag_history'].append({'user': user_input, 'time': request.json.get('time')})
    state = session.get('rag_state', 'INITIAL')
    ai_response = "Произошла непредвиденная ошибка. Состояние сброшено."

    def reset_state():
        keys_to_pop = ['rag_state', 'rag_found_docs', 'rag_original_question', 'rag_selected_doc_name', 'rag_subtopics']
        for key in keys_to_pop: session.pop(key, None)
        session['rag_state'] = 'INITIAL'

    if not summary_vector_store or not full_vector_store:
        ai_response = "База данных документов не загружена."
    
    elif state == 'INITIAL':
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
            retriever = full_vector_store.as_retriever(search_kwargs={'k': 5, 'filter': {'source': selected_doc_name}})
            relevant_chunks = retriever.invoke(original_question)
            if not relevant_chunks:
                ai_response = f"В документе '{selected_doc_name}' не найдено релевантных фрагментов."
                reset_state()
            else:
                context_text = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
                # --- УЛУЧШЕНИЕ №2: ПРОДВИНУТЫЙ ПРОМПТ ДЛЯ ТЕМ ---
                subtopic_prompt = ChatPromptTemplate.from_template("""Твоя задача — помочь пользователю найти информацию. Проанализируй текст и вопрос. Выдели 3-4 ключевые, осмысленные темы из текста, которые наиболее релевантны вопросу. Темы должны быть короткими и понятными, как заголовки разделов.

**Правила:**
- Извлекай только самые важные темы.
- Не выдумывай ничего, чего нет в тексте.
- Ответь ТОЛЬКО списком тем, каждая на новой строке.

**Пример ответа:**
Согласование нестандартного договора
Проведение предстрахового осмотра
Принятие андеррайтингового решения

**Контекст:**
{context}

**Вопрос пользователя:** {question}""")
                subtopic_chain = subtopic_prompt | llm
                subtopics_obj = subtopic_chain.invoke({"context": context_text, "question": original_question})
                subtopics = [topic.strip() for topic in subtopics_obj.content.split('\n') if topic.strip()]
                session.update({'rag_state': 'AWAITING_SUBTOPIC_CONFIRMATION', 'rag_selected_doc_name': selected_doc_name, 'rag_subtopics': subtopics})
                response_lines = [f"Отлично. В документе '{selected_doc_name}' я нашел следующие темы:", ""]
                for i, topic in enumerate(subtopics): response_lines.append(f"{i+1}. {topic}")
                response_lines.append("\nКакая тема вас интересует больше всего? Укажите номер или название.")
                ai_response = "\n".join(response_lines)
        else:
            ai_response = "Не удалось распознать ваш выбор. Пожалуйста, попробуйте еще раз."

    elif state == 'AWAITING_SUBTOPIC_CONFIRMATION':
        subtopics, selected_topic_str = session.get('rag_subtopics', []), None
        try:
            choice_index = int(user_input.strip()) - 1
            if 0 <= choice_index < len(subtopics): selected_topic_str = subtopics[choice_index]
        except (ValueError, IndexError):
            for topic in subtopics:
                if user_input.lower().strip() in topic.lower(): selected_topic_str = topic; break
        
        if selected_topic_str:
            selected_doc_name = session['rag_selected_doc_name']
            final_query = session['rag_original_question'] + " " + selected_topic_str
            retriever = full_vector_store.as_retriever(search_kwargs={'k': 5, 'filter': {'source': selected_doc_name}})
            final_chunks = retriever.invoke(final_query)
            final_answer_prompt = ChatPromptTemplate.from_template("""Ты — ассистент-аналитик. Дай точный ответ на вопрос пользователя, основываясь ИСКЛЮЧИТЕЛЬНО на предоставленном контексте.

**Правила:**
1. Ответ должен быть СТРОГО в рамках контекста.
2. Если ответа нет, напиши только: 'В документе "{doc_name}" нет информации по этому вопросу.'
3. Структурируй ответ: **Ответ:**, **Цитата из документа:**, **Источник:**.

**Контекст из документа "{doc_name}":**
{context}

**Вопрос пользователя:** {input}""")
            answer_chain = create_stuff_documents_chain(llm, final_answer_prompt)
            response = answer_chain.invoke({"input": session['rag_original_question'], "doc_name": selected_doc_name, "context": final_chunks})
            ai_response = response
            download_link = url_for('download_file', filename=selected_doc_name)
            ai_response += f'\n\n[Скачать документ]({download_link})'
            reset_state()
        else:
             ai_response = "Не удалось распознать ваш выбор темы. Пожалуйста, укажите номер или название из списка."

    session['rag_history'].append({'ai': ai_response, 'time': request.json.get('time')})
    session.modified = True
    return jsonify({'answer': ai_response})


# --- Маршруты для Общего чата и утилиты (без изменений) ---
@app.route('/general')
def general_chat_page():
    # ... (код без изменений) ...
    if 'general_history' not in session:
        session['general_history'] = []
    return render_template('index.html', history=session['general_history'], title="Общий AI Ассистент", api_url=url_for('ask_general'))


@app.route('/ask_general', methods=['POST'])
def ask_general():
    # ... (код без изменений) ...
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
    # ... (код без изменений) ...
    return send_from_directory(DOCUMENTS_FOLDER, filename, as_attachment=True)


@app.route('/reset')
def reset():
    # ... (код без изменений) ...
    referrer = request.referrer
    if referrer and 'general' in referrer:
        session.pop('general_history', None)
        return redirect(url_for('general_chat_page'))
    else:
        # Сбрасываем состояние RAG чата
        keys_to_pop = ['rag_history', 'rag_state', 'rag_found_docs', 'rag_original_question', 'rag_selected_doc_name', 'rag_subtopics']
        for key in keys_to_pop:
            session.pop(key, None)
        return redirect(url_for('rag_chat_page'))


if __name__ == '__main__':
    build_vector_stores()
    app.run(host='0.0.0', port=5000, debug=True)
