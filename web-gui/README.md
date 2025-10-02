## Структура проекта

```
realtime-dashboard/
├── app/
│   ├── main.py              # FastAPI сервер
│   ├── templates/
│   │   └── index.html       # HTML разметка
│   └── static/
│       ├── css/
│       │   └── style.css    # Стили
│       └── js/
│           └── charts.js    # JavaScript для графиков
├── Dockerfile               # Docker образ
├── docker-compose.yml       # Оркестрация
├── requirements.txt         # Python зависимости
└── README.md               # Документация
```

## Запуск

### Вариант 1: Docker

```bash
docker-compose up --build

# http://localhost:8000
```

### Вариант 2: Локальный запуск

```bash
pip install -r requirements.txt

cd app && python main.py

# http://localhost:8000
```