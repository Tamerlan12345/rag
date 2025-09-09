import os
from flask import Flask, request, render_template

# Импортируем только то, что нужно для прямого чата
from langchain_community.chat_models import ChatOllama

# --- Инициализация Flask приложения ---
app = Flask(__name__)

# --- Главная страница ---
@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    # Если пользователь отправил сообщение (форму)
    if request.method == 'POST':
        # Получаем вопрос пользователя из формы
        question = request.form['question']
        
        # --- Прямое общение с моделью ---
        # 1. Определяем модель
        llm = ChatOllama(model="phi3")

        # 2. Отправляем запрос и получаем ответ
        answer = llm.invoke(question).content

    return render_template('index.html', answer=answer)

# --- Запуск приложения ---
if __name__ == '__main__':
    app.run(host='0.0.0', port=5000, debug=True)
