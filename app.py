import os
from flask import Flask, request, render_template, session, url_for, send_from_directory, jsonify
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
app.secret_key = 'ultra_robust_rag_agent_secret'

DOCUMENTS_FOLDER = 'documents'
VECTOR_STORE_SUMMARIES_PATH = 'vector_store_summaries'
VECTOR_STORE_FULL_PATH = 'vector_store_full'
os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

# Инициализируем модель с увеличенным тайм-аутом для надежности
llm = ChatOllama(model="phi3:mini", timeout=300)

print("Инициализация модели для эмбеддингов...")
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

summary_vector_store = None
full_vector_store = None

def build_vector_stores():
    """Создает или загружает две векторные базы: для названий и для полного текста."""
    global summary_vector_store, full_vector_store
    if os.path.exists(VECTOR_STORE_SUMMARIES_PATH) and os.path.exists(VECTOR_STORE_FULL_PATH):
        print("Загрузка существующих векторных баз...")
        summary_vector_store = Chroma(persist_directory=VECTOR_STORE_SUMMARIES_PATH, embedding_function=embeddings)
        full_vector_store = Chroma(persist_directory=VECTOR_STORE_FULL_PATH, embedding_function=embeddings)
    else:
        print("Создание новых векторных баз...")
        summaries, full_docs_splits = [], []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        file_list = os.listdir(DOCUMENTS_FOLDER)
        for filename in tqdm(file_list, desc="Обработка документов"):
            filepath = os.path.join(DOCUMENTS_FOLDER, filename)
            try:
                if filename.endswith('.pdf'): loader = PyPDFLoader(filepath)
                elif filename.endswith('.docx'): loader = Docx2txtLoader(filepath)
                else: print(f"\nПропускаем неподдерживаемый файл: {filename}"); continue
                
                docs = loader.load()
                if not docs: print(f"\nНе удалось извлечь текст из файла: {filename}"); continue

                summary_doc = docs[0].model_copy()
                summary_doc.page_content = f"Название документа: {filename}"
                summary_doc.metadata = {"source": filename}
                summaries.append(summary_doc)

                splits = text_splitter.split_documents(docs)
                for split in splits: split.metadata["source"] = filename
                full_docs_splits.extend(splits)
            except Exception as e:
                print(f"\nНе удалось обработать файл {filename}: {e}")

        if not summaries or not full_docs_splits:
            print("Не найдено подходящих документов для индексации."); return
            
        print("Создание векторной базы для названий..."); summary_vector_store = Chroma.from_documents(documents=summaries, embedding=embeddings, persist_directory=VECTOR_STORE_SUMMARIES_PATH)
        print("Создание векторной базы для полного текста..."); full_vector_store = Chroma.from_documents(documents=full_docs_splits, embedding=embeddings, persist_directory=VECTOR_STORE_FULL_PATH)
        
    print("Векторные базы успешно созданы и готовы к работе.")

@app.route('/')
def index():
    if 'history' not in session:
        session['history'], session['state'] = [], 'INITIAL'
    return render_template('index.html', history=session['history'])

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(DOCUMENTS_FOLDER, filename, as_attachment=True)

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get('question')
    session['history'].append({'user': user_input, 'time': request.json.get('time')})
    state = session.get('state', 'INITIAL')
    ai_response = ""

    # Функция сброса состояния для нового диалога
    def reset_state():
        session['state'] = 'INITIAL'
        session.pop('found_docs', None)
        session.pop('original_question', None)

    if not summary_vector_store or not full_vector_store:
        ai_response = "База данных документов не загружена. Проверьте консоль на наличие ошибок при запуске."
    elif state == 'INITIAL':
        retriever = summary_vector_store.as_retriever(search_kwargs={"k": 3})
        found_docs = retriever.invoke(user_input)
        doc_filenames = sorted(list(set([doc.metadata['source'] for doc in found_docs])))
        if not doc_filenames:
            ai_response = "К сожалению, я не нашел подходящих документов по вашему запросу."
        else:
            session.update({'state': 'AWAITING_CONFIRMATION', 'found_docs': doc_filenames, 'original_question': user_input})
            response_lines = ["Я нашел несколько потенциально подходящих документов. В каком из них мне искать ответ?", ""]
            for i, name in enumerate(doc_filenames): response_lines.append(f"{i+1}. {name}")
            response_lines.append("\nПожалуйста, укажите номер или название документа.")
            ai_response = "\n".join(response_lines)
    elif state == 'AWAITING_CONFIRMATION':
        found_docs, selected_doc_name = session.get('found_docs', []), None
        try:
            choice_index = int(user_input.strip()) - 1
            if 0 <= choice_index < len(found_docs): selected_doc_name = found_docs[choice_index]
        except (ValueError, IndexError):
            for name in found_docs:
                if user_input.lower().strip() in name.lower(): selected_doc_name = name; break
        
        if selected_doc_name:
            try:
                original_question = session['original_question']
                # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Сначала находим релевантные части, потом отправляем в модель
                retriever = full_vector_store.as_retriever(search_kwargs={'k': 5, 'filter': {'source': selected_doc_name}})
                relevant_chunks = retriever.invoke(original_question)

                if not relevant_chunks:
                    ai_response = f"В документе '{selected_doc_name}' не найдено релевантных фрагментов по вашему запросу."
                else:
                    final_answer_prompt = ChatPromptTemplate.from_template("""Ты — ассистент-аналитик. Твоя задача — дать точный и исчерпывающий ответ на вопрос пользователя, основываясь ИСКЛЮЧИТЕЛЬНО на предоставленном ниже контексте из документа.
**Правила ответа:**
1. Твой ответ должен быть СТРОГО в рамках предоставленного контекста. Не добавляй никакой информации извне.
2. Если контекст не содержит ответа на вопрос, напиши только: 'В документе "{doc_name}" нет информации по этому вопросу.'
3. Структурируй свой ответ следующим образом:
**Ответ:** [Здесь твой прямой и краткий ответ на вопрос]
**Цитата из документа:**
> [Здесь дословная цитата из контекста, на которой основан твой ответ]
**Источник:** [Здесь название документа '{doc_name}' и, если возможно, номер пункта или шага из цитаты]
---
**Контекст из документа "{doc_name}":**
{context}
---
**Вопрос пользователя:** {input}""")
                    
                    answer_chain = create_stuff_documents_chain(llm, final_answer_prompt)
                    # Передаем в модель ТОЛЬКО релевантные фрагменты
                    response = answer_chain.invoke({"input": original_question, "doc_name": selected_doc_name, "context": relevant_chunks})
                    ai_response = response
                    download_link = url_for('download_file', filename=selected_doc_name)
                    ai_response += f'\n\n[Скачать документ]({download_link})'
                
                reset_state()
            except Exception as e:
                print(f"ОШИБКА при вызове LLM: {e}")
                ai_response = "Извините, модель не смогла обработать запрос. Возможно, стоит попробовать другой документ или переформулировать вопрос."
                reset_state()
        else:
            ai_response = "Не удалось распознать ваш выбор. Пожалуйста, укажите номер или название документа из списка."

    session['history'].append({'ai': ai_response, 'time': request.json.get('time')})
    session.modified = True
    return jsonify({'answer': ai_response})

if __name__ == '__main__':
    build_vector_stores()
    app.run(host='0.0.0', port=5000, debug=True)
