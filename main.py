from argparse import ArgumentParser
from source.data_manipulation import DataLoader
from source.recognizer import VoiceRecognizer
import json
import os, sys
import numpy as np
import subprocess
from pathlib import Path
import requests
import shutil

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

model_link = "https://huggingface.co/openai/whisper-large-v3"
model_path = data_loader.MAIN_FOLDER / 'model'

if not model_path.exists():

    class ScriptException(Exception):
        def __init__(self, returncode, stdout, stderr, script):
            self.returncode = returncode
            self.stdout = stdout.decode('utf-8') if stdout else ''
            self.stderr = stderr.decode('utf-8') if stderr else ''
            self.script = script
            super().__init__(f'Ошибка в скрипте: {script}\nВозвращенный код: {returncode}\nStdout: {self.stdout}\nStderr: {self.stderr}')

    class FIRST_INIT_BASH_SCRIPT:
        @staticmethod
        def run_bash_commands(scripts: list, stdin=None) -> None:
            for script in scripts:
                proc = subprocess.Popen(['bash', '-c', script], 
                                        stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE,
                                        stdin=subprocess.PIPE if stdin else None)
                
                stdout, stderr = proc.communicate()

                if proc.returncode:
                    raise ScriptException(proc.returncode, stdout, stderr, script)

                print(stdout.decode())
                print(stderr.decode())

    def check_and_install_git_lfs():
        try:
            proc = subprocess.Popen(['git', 'lfs'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = proc.communicate()

            if b"'lfs' is not a git command" in stderr:
                print("Git LFS не установлен. Устанавливаем...")

                install_lfs_script = 'curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash'
                
                proc_install_script = subprocess.Popen(['bash', '-c', install_lfs_script], 
                                                    stdout=subprocess.PIPE, 
                                                    stderr=subprocess.PIPE)
                stdout_script, stderr_script = proc_install_script.communicate()

                if proc_install_script.returncode:
                    raise ScriptException(proc_install_script.returncode, stdout_script, stderr_script, install_lfs_script)
                

                apt_install_git_lfs = 'sudo apt-get install -y git-lfs'
                
                proc_install_lfs = subprocess.Popen(['bash', '-c', apt_install_git_lfs], 
                                                    stdout=subprocess.PIPE, 
                                                    stderr=subprocess.PIPE)
                stdout_lfs, stderr_lfs = proc_install_lfs.communicate()

                if proc_install_lfs.returncode:
                    raise ScriptException(proc_install_lfs.returncode, stdout_lfs, stderr_lfs, apt_install_git_lfs)
                else:
                    print("Git LFS успешно установлен, продолжаем...")

        except Exception as e:
            print(f"Произошла ошибка при попытке установить Git LFS: {e}") 

    if requests.get('https://ya.ru').ok:
        try:
            check_and_install_git_lfs()

            bash_script_list = [
                'git lfs install',
                'ls -l',
                f'git clone https://huggingface.co/openai/whisper-large-v3 {model_path}'
            ]
            
            FIRST_INIT_BASH_SCRIPT.run_bash_commands(bash_script_list)

        except ScriptException as e:
            print(f"Ошибка: {e}")
else:
    pass

# TODO: // Загрузка модели для распознавания речи (русской английской).
# TODO: // Инициализация модели: проверка по памяти 
# TODO: // Использование модели и логгирование в жсон. 
