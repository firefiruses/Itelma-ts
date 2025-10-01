import asyncio
import pandas as pd
from datetime import datetime
import logging
from app.data_loader import csv_path_to_dataframe
import json
import websockets

logger = logging.getLogger(__name__)

class CTGDataEmulator:
    """
    Класс для эмуляции потоковых данных КТГ.
    Загружает данные из CSV и воспроизводит их в реальном времени.
    """
    def __init__(self, bpm_df, uc_df, time_scale=1.0):
        """
        Args:
            bpm_csv_path (str): Путь к CSV с данными сердцебиения плода.
            uc_csv_path (str): Путь к CSV с данными сокращений матки.
            time_scale (float): Масштаб времени. 1.0 = реальное время, 2.0 = в 2 раза быстрее.
        """
        # Загружаем данные
        self.bpm_data = bpm_df
        self.uc_data = uc_df
        
        # Проверяем, что временные столбцы существуют
        if not all(col in self.bpm_data.columns for col in ['time_sec', 'value']):
            raise ValueError("bpm CSV must have 'time_sec' and 'value' columns")
        if not all(col in self.uc_data.columns for col in ['time_sec', 'value']):
            raise ValueError("UC CSV must have 'time_sec' and 'value' columns")
            
        self.time_scale = time_scale
        self.is_playing = False

        logger.info(f"Эмулятор инициализирован")
        
    async def start_streaming(self, websocket, start_time=0.0, delay = 0.2):
        """
        Начинает потоковую передачу данных через WebSocket.
        
        Args:
            websocket: Объект WebSocket соединения.
            start_time (float): Время с которого начать воспроизведение (в секундах).
        """
        self.is_playing = True
        
        # Находим начальную точку в данных
        bpm_index = self.bpm_data[self.bpm_data['time_sec'] >= start_time].index.min()
        uc_index = self.uc_data[self.uc_data['time_sec'] >= start_time].index.min()
        # Если индексы не найдены (например, start_time больше максимального времени в данных)
        if pd.isna(bpm_index) or pd.isna(uc_index):
            logger.warning(f"Start time {start_time} выходит за пределы данных")
            await websocket.send(json.dumps({
                "status": "error", 
                "message": f"Start time {start_time} beyond available data"
            }))
            return
        
        logger.info(f"Начало потоковой передачи с времени {start_time}")
        

        bpm_row = self.bpm_data.iloc[bpm_index]
        uc_row = self.uc_data.iloc[uc_index]
        bpm_time = bpm_row['time_sec']
        uc_time = uc_row['time_sec']


        t = start_time
        try:
            while bpm_index < len(self.bpm_data) or uc_index < len(self.uc_data):
                if not self.is_playing:
                    logger.info("Потоковая передача остановлена")
                    break
                if bpm_time <= uc_time:
                    await asyncio.sleep((bpm_time-t)/self.time_scale)
                    t =bpm_time
                    data_packet = {
                        "timestamp": datetime.now().isoformat(), # Метка времени получения данных
                        "time": t,
                        "type": "bpm",  # Сердцебиение плода
                        "bpm": bpm_row['value']     # Сокращения матки
                    }
                    bpm_index += 1
                    if bpm_index < len(self.bpm_data):
                        bpm_row = self.bpm_data.iloc[bpm_index]
                        bpm_time = bpm_row['time_sec']
                    else:
                        bpm_time = 1000000000
                else:
                    await asyncio.sleep((uc_time-t)/self.time_scale)
                    t =uc_time
                    data_packet = {
                        "timestamp": datetime.now().isoformat(), # Метка времени получения данных
                        "time": t,
                        "type": "uc",  # Сердцебиение плода
                        "uc": bpm_row['value']     # Сокращения матки
                    }
                    uc_index += 1
                    if uc_index < len(self.uc_data):
                        uc_row = self.uc_data.iloc[uc_index]
                        uc_time = uc_row['time_sec']
                    else:
                        uc_time = 1000000000
                try:
                    await websocket.send(json.dumps(data_packet))
                    print(f"Sent: {data_packet}")
                except websockets.exceptions.ConnectionClosed:
                    print("Client disconnected.")
                    break
        except Exception as e:
            logger.error(f"Ошибка во время потоковой передачи: {e}")
            await websocket.send(json.dumps({
                "status": "error", 
                "message": f"Streaming error: {str(e)}"
            }))