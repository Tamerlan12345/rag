import os
from flask import Flask, request, render_template, jsonify

# Импортируем только то, что нужно для прямого чата
from langchain_community.chat_models import ChatOllama

# --- Инициализация Flask приложения ---
app = Flask(__name__)

# --- Инициализация модели ---
# Мы создаем один экземпляр модели, чтобы не загружать ее при каждом запросе
llm = ChatOllama(model="qwen2:1.5b")

# --- Главная страница для чата в браузере (остаётся как есть) ---
@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    if request.method == 'POST':
        question = request.form['question']
        # Используем уже созданный экземпляр модели
        answer = llm.invoke(question).content
    return render_template('index.html', answer=answer)

# --- НОВЫЙ ЭНДПОИНТ ДЛЯ ВЕБХУКА ---
@app.route('/webhook', methods=['POST'])
def webhook():
    # Проверяем, что нам пришли данные в формате JSON
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    # Получаем JSON из тела запроса
    data = request.get_json()

    # Проверяем, есть ли в данных ключ "message"
    if 'message' not in data:
        return jsonify({"error": "Missing 'message' key in JSON payload"}), 400

    # Получаем сообщение
    user_message = data['message']

    print(f"Получено сообщение через вебхук: {user_message}") # Логируем в терминал

    # --- Отправляем сообщение модели и получаем ответ ---
    try:
        # Используем уже созданный экземпляр модели
        model_response = llm.invoke(user_message).content
        print(f"Ответ модели: {model_response}") # Логируем ответ

        # Возвращаем успешный ответ в формате JSON
        return jsonify({"status": "success", "response": model_response})
    
    except Exception as e:
        print(f"Ошибка при вызове модели: {e}")
        return jsonify({"error": "Failed to get response from LLM"}), 500

# --- Запуск приложения ---
if __name__ == '__main__':
    app.run(host='0.0.0', port=5000, debug=True)
