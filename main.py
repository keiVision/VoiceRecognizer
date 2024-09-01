from argparse import ArgumentParser
from source.data_manipulation import DataLoader
from source.recognizer import VoiceRecognizer
import os, sys, json
import numpy as np
import subprocess
from pathlib import Path
import requests

class ScriptException(Exception):
    def __init__(self, returncode, stdout, stderr, script):
        self.returncode = returncode
        self.stdout = stdout.decode('utf-8') if stdout else ''
        self.stderr = stderr.decode('utf-8') if stderr else ''
        self.script = script
        super().__init__(f'Ошибка в скрипте: {script}\nВозвращенный код: {returncode}\nStdout: {self.stdout}\nStderr: {self.stderr}')

parser = ArgumentParser()
data_loader = DataLoader()

model_path = Path(data_loader.MAIN_FOLDER) / 'model' / 'whisper-base'
voice_recognizer = VoiceRecognizer(model_path)

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
    
if requests.get('https://ya.ru').ok:
    if not model_path.exists():

        try:
            proc = subprocess.Popen(['bash', 'init_bash.sh'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = proc.communicate()

            print(stdout.decode())
            print(stderr.decode())

            if proc.returncode:
                raise ScriptException(proc.returncode, stdout, stderr, 'init_bash.sh')

        except ScriptException as e:
            print(f"Ошибка: {e}")
else:
    print("Нет подключения к интернету.")

print(voice_recognizer.process_sound(sound_vector=sound_vector))

# TODO: // Инициализация модели: проверка по памяти 
# TODO: // Использование модели и логгирование в жсон. 