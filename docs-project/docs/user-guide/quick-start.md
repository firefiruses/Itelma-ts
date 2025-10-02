<p>Быстрый старт (Docker)</p>
<p>1. Клонирование репозитория</p>
<p>bash</p>
<p>git clone https://github.com/firefiruses/itelma-ts.git</p>
<p>cd itelma-ts</p>
<p>2. Настройка переменных окружения</p>
<p>Создайте файл .env в корне проекта:</p>
<br>
<p>env</p>
<p># Базовые настройки</p>
<p>API_HOST=0.0.0.0</p>
<p>API_PORT=8000</p>
<p>WS_SERVER_HOST=ws-server</p>
<p>WS_SERVER_PORT=8765</p>
<br>
<p>3. Запуск системы</p>
<p>bash</p>
<p># Запуск всех сервисов</p>
<p>docker-compose up -d</p>
<br>
<p># Проверка статуса</p>
<p>docker-compose ps</p>
<p>4. Проверка работоспособности</p>
<p>bash</p>
<p># Проверка API</p>
<p>curl http://localhost:8000/health</p>
<br>
<p># Проверка WebSocket соединения</p>
<p>curl http://localhost:8000/stream/bpm</p>
