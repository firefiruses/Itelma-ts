import pandas as pd
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def csv_path_to_dataframe(patient_type, folder_number):
    patient_path = Path(f"data/{patient_type}/{folder_number}")

    # Проверяем существование основной папки
    if not patient_path.exists():
        raise FileNotFoundError(f"Папка пациента не найдена: {folder_number}")
    
    # Пути к подпапкам
    bpm_folder = patient_path / "bpm"
    uterus_folder = patient_path / "uterus"

    # Проверяем существование подпапок
    if not bpm_folder.exists():
        raise FileNotFoundError(f"Папка с данными сердцебиения не найдена: {bpm_folder}")
    if not uterus_folder.exists():
        raise FileNotFoundError(f"Папка с данными сокращений матки не найдена: {uterus_folder}")
    
    def load_and_combine_csv_files(folder_path):
        # Ищем все CSV файлы в папке
        csv_files = sorted(folder_path.glob("*.csv"))
        
        if not csv_files:
            raise ValueError(f"В папке {folder_path} не найдено CSV файлов")
        
        logger.info(f"Найдено {len(csv_files)} файлов в {folder_path.name}")

        
        # Загружаем и объединяем все файлы
        dataframes = []
        current_time_offset = 0
        
        for file_path in csv_files:
            # Если это не первый файл, добавляем временное смещение
            if dataframes:
                # Находим максимальное время в предыдущем файле и добавляем небольшой зазор
                last_max_time = df['time_sec'].iloc[-1]
                current_time_offset += last_max_time + 0.1  # Добавляем 100 мс зазор между файлами
            try:         
                # Загружаем CSV файл
                df = pd.read_csv(file_path)

                # Применяем временное смещение
                df['time_sec'] = df['time_sec'] + current_time_offset
                dataframes.append(df)

                logger.debug(f"Загружен {file_path.name} - {len(df)} записей")
            except Exception as e:
                logger.error(f"Ошибка при загрузке файла {file_path.name}: {e}")
                continue
        
        # Объединяем все DataFrame
        combined_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Объединено {len(dataframes)} файлов. Общее количество точек: {len(combined_df)}")
        
        return combined_df
    
    # Загружаем данные сердцебиения плода и сокращений матки
    bpm_df = load_and_combine_csv_files(bpm_folder)
    uc_df = load_and_combine_csv_files(uterus_folder)
    
    logger.info(f"Сердцебиение плода: {len(bpm_df)} записей")
    logger.info(f"Сокращения матки: {len(uc_df)} записей")
    
    return bpm_df, uc_df