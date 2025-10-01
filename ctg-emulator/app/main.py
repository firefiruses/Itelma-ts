import asyncio
import websockets
import json
import logging
from pathlib import Path
from app.ctg_emulator import CTGDataEmulator
from app.data_loader import csv_path_to_dataframe

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

emulators = {}

async def handle_connection(websocket):
    """
    Обрабатывает входящие WebSocket соединения.
    """
    session_id = f"{websocket.remote_address}_{id(websocket)}"
    logger.info(f"Новое подключение: {session_id}")
    
    try:
        # Ожидаем команды от клиента
        async for message in websocket:
            command = json.loads(message)
            action = command.get("action")
            logger.info(f"Получена команда '{action}' от {session_id}")
            
            if action == "start":
                # Загружаем данные для конкретного пациента (можно передавать ID пациента в команде)
                patient_type, folder_number = command.get("patient_type"), command.get("folder_number")
                bpm_df, uc_df = csv_path_to_dataframe(patient_type, folder_number)
                
                if session_id in emulators:
                        emulators[session_id].is_playing = False

                # Создаем или пересоздаем эмулятор для выбранного пациента
                try:
                    emulator = CTGDataEmulator(bpm_df, uc_df)
                    start_time = command.get("start_time", 0.0)
                    
                    # Запускаем потоковую передачу в отдельной задаче
                    asyncio.create_task(emulator.start_streaming(websocket, start_time))
                    await websocket.send(json.dumps({"status": "started", "folder_number": folder_number}))
                    
                except FileNotFoundError:
                    error_msg = f"Data files for {patient_type} patient {folder_number} not found."
                    await websocket.send(json.dumps({"status": "error", "message": error_msg}))
                    logger.error(error_msg)
                    
            elif action == "stop":
                if session_id in emulators:
                    emulators[session_id].is_playing = False
                    await websocket.send(json.dumps({"status": "stopped"}))
                    logger.info(f"Эмулятор остановлен для {session_id}")

            else:
                await websocket.send(json.dumps({
                    "status": "error", 
                    "message": f"Неизвестная команда: {action}"
                }))
                    

                    
    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")
    finally:
        # Останавливаем эмулятор при разрыве соединения
        if session_id in emulators:
            emulators[session_id].is_playing = False
            del emulators[session_id]


async def health_check(websocket, path):
    """
    Простой health-check endpoint
    """
    await websocket.send(json.dumps({"status": "healthy", "service": "CTG Emulator"}))


async def main():
    """
    Запускает WebSocket сервер.
    """
    host = "0.0.0.0"  # Слушаем все интерфейсы
    port = 8765
    
    # Запускаем основной сервер
    main_server = await websockets.serve(
        handle_connection, 
        host, 
        port
    )
    
    # Health check endpoint
    health_server = await websockets.serve(
        health_check,
        host,
        8766  # Отдельный порт для health check
    )
    
    logger.info(f"CTG Emulator WebSocket сервер запущен на ws://{host}:{port}")
    logger.info(f"Health check доступен на ws://{host}:8766")
    
    # Запускаем сервера на постоянную работу
    await asyncio.gather(
        main_server.wait_closed(),
        health_server.wait_closed()
    )

if __name__ == "__main__":
    asyncio.run(main())