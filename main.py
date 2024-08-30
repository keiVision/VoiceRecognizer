from argparse import ArgumentParser
from source.data_manipulation import DataLoader
from source.recognizer import VoiceRecognizer
import json
import os, sys
import numpy as np

parser = ArgumentParser(description="Аудио обработка")
data_loader = DataLoader()
voice_recognizer = VoiceRecognizer()

parser.add_argument('--file',
                    help='Путь к файлу, над которым будет работать алгоритм.',
                    type=str,
                    required=True)

parser.add_argument('--volume',
                    help='Настроить громкость звуковой дорожки (например, 1.0 для исходного уровня, 0.5 для уменьшения громкости вдвое).',
                    type=float,
                    required=False)

parser.add_argument('--speed',
                    help='Изменить скорость воспроизведения звуковой дорожки (например, 1.0 для исходной скорости, 2.0 для увеличения скорости вдвое).',
                    type=float,
                    required=False)

args = parser.parse_args()

sound_vector, sample_rate = data_loader.load_sound_file(args.file)

if args.volume is not None:
    sound_vector, sample_rate = data_loader.volume_control(
        sound_vector=sound_vector, 
        sample_rate=sample_rate, 
        volume_lvl=args.volume
    )

if args.speed is not None:
    sound_vector, sample_rate = data_loader.sound_speed_control(
        sound_vector=sound_vector, 
        sample_rate=sample_rate, 
        speed_lvl=args.speed
    )

# TODO: // Подготовка к загрузке модели: проверки наличия моделей в путях, проверка доступа к интернету
# TODO: // Загрузка модели для распознавания речи (русской английской).
# TODO: // Инициализация модели: проверка по памяти 
# TODO: // Использование модели и логгирование в жсон. 
