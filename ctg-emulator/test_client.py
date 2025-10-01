import asyncio
import websockets
import json
import sys

async def test_emulator():
    try:
        print("Подключаемся к WebSocket серверу...")
        async with websockets.connect("ws://localhost:8765") as websocket:
            
            # Тест 2: Запуск эмулятора
            print("\n2. Запуск эмулятора...")
            start_command = {
            "action": "start",
            "patient_type": "hypoxia",
            "folder_number": "1",
            "start_time": 0.0
            }
            await websocket.send(json.dumps(start_command))
            
            # Получаем подтверждение запуска
            response = await websocket.recv()
            print(f"Статус запуска: {response}")
            
            # Тест 3: Получение данных в реальном времени
            print("\n3. Получение данных... (Ctrl+C для остановки)")
            count = 0
            try:
                async for message in websocket:
                    data = json.loads(message)
                    
                    print(data)
                    count += 1
                    if count >= 10:  # Получаем 10 записей и останавливаемся
                        print("Получено 10 записей, останавливаем...")
                        await websocket.send(json.dumps({"action": "stop"}))
                        break
            except asyncio.CancelledError:
                pass
                
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    asyncio.run(test_emulator())