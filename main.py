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

def check_internet_connection(url="http://www.google.com"):
    """
    Проверяет подключение к интернету.
    Возвращает True, если подключение есть, и False, если нет.
    """
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    
    except requests.ConnectionError:
        return False
    
def log_to_json(result, output_file="recognized_log.json"):
    """
    Логирует результат в формате JSON и сохраняет в файл.
    :param result: Словарь с результатом расшифровки.
    :param output_file: Имя файла, в который будет записан результат.
    """
    with open(output_file, 'a', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=4)
        file.write('\n') 

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

parser.add_argument('--language',
                    help="Возможность вручную указать язык 'ru' или 'en' для распознавания (по-умолчанию: не указан и определяется автоматически).",
                    type=str,
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
    
if check_internet_connection():
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
    pass

transcription_result = voice_recognizer.process_sound(sound_vector=sound_vector, language=args.language)

print(transcription_result)

log_to_json({"file": args.file, "transcription": transcription_result}, output_file="recognized_log.json")
