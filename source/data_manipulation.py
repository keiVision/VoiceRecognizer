import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple


class DataLoader:
    def __init__(self, main_folder: Path = None,
                       is_plot: bool = False) -> None:
        """
        Инициализация DataLoader.

        :param main_folder: Основная папка проекта. Создается автоматически.
        :param is_plot: Флаг для отображения графиков при загрузке звукового файла.
        """
        self.MAIN_FOLDER = Path(main_folder or Path(__file__).resolve().parent.parent)
        self.data_folder = self.MAIN_FOLDER / 'data'
        self.is_plot = is_plot
        
        if not self.data_folder.exists():
            self.data_folder.mkdir(parents=True)


    def load_sound_file(self, file_name: str, 
                              info: bool = False) -> Tuple[int, np.ndarray]:
        """
        Загружает звуковой файл формата WAV и опционально отображает информацию и график.

        :param file_name: Имя звукового файла, который нужно загрузить.
        :param info: Флаг(опционально) для отображения информации о частоте дескретизации и размерности вектора.
        :return: Кортеж, содержащий частоту дискретизации и звуковой вектор.
        """

        if file_name is None:
            raise ValueError("Имя файла не должно быть None.")
        
        file_dir = self.data_folder / file_name

        if not file_dir.exists():
            raise FileNotFoundError(f"Файл '{file_dir}' не найден. Находится-ли {file_name} в папке {self.data_folder} ?")
        
        if not isinstance(file_name, str): 
            raise TypeError(f"Ожидается тип 'str' для имени файла, но получено: {type(file_name).__name__}")

        sound_vector, sample_rate = librosa.load(file_dir)

        if info:
            print(f"Sound sample rate is: {sample_rate}")
            print(f"Sound vector shape is: {sound_vector.shape}")

        if self.is_plot:
            self._plot_sound_wave(sample_rate, sound_vector)

        return sound_vector, sample_rate


    def write_wav(self, file_name:str, 
                        sound_array: np.ndarray, 
                        sample_rate:int = 22050) -> None:
        """
        Сохраняет переданный звуковой вектор в .wav файл.
        :param file_name: Передается название файла
        :param sound_array: Передается звуковой вектор
        :param sample_rate: Передается частота дискретизации (по-умолчанию 22050)
        """
        if not isinstance(file_name, str) or file_name is None:
            raise ValueError('Название файла не является строкой или не задано.')
        
        if not isinstance(sound_array, np.ndarray):
            raise ValueError('Звуковой вектор должен быть numpy массивом.')
        
        if not isinstance(sample_rate, int):
            raise ValueError('Частота дискретизации должна быть целым числом.')
        
        if not file_name.lower().endswith('.wav'):
            raise ValueError('Название файла должно оканчиваться на ".wav".')

        sf.write(file_name, sound_array, sample_rate)


    def volume_control(self, sound_vector: np.ndarray, 
                             sample_rate: int = 22050, 
                             volume_lvl: float = 1.0) -> Tuple[np.ndarray, int]:
        """
        Позволяет регулировать громкость звуковой дорожки.
        :param sound_vector: Звуковой вектор над которым выполняется действие.
        :param sample_rate: Частота дискретизации относящаяся к вектору sound_vector (по-умолчанию: 22050).
        :return processed_sound_vector, sample_rate: Обработанный звуковой файл.
        """
        sound_vector *= volume_lvl
        return sound_vector, sample_rate


    def sound_speed_control(self, sound_vector: np.ndarray,
                                  sample_rate: int = 22050,
                                  speed_lvl: float = 1.0, 
                                  is_shift: bool = False) -> Tuple[np.ndarray, int]:
        """
        Позволяет регулировать скорость воспроизведения звуковой дорожки.
        :param sound_vector: Звуковой вектор над которым выполняется действие.
        :param sample_rate: Частота дискретизации относящаяся к вектору sound_vector (по-умолчанию: 22050).
        :param is_shift: Флаг, позволяющий использовать питч тональности измененной дорожки (по-умолчанию False) 
        :return processed_vector, sample_rate: Обработанный звуковой файл.
        """
        edited_sound_vector = librosa.effects.time_stretch(y=sound_vector, rate=speed_lvl)

        if not is_shift:
            return edited_sound_vector, sample_rate
        
        else:
            if speed_lvl > 1.0: 
                n_steps = (speed_lvl * 10) - 10 # Рассчитывается пропорционально уровню повышения скорости
            
            elif speed_lvl < 1.0: 
                n_steps = -(10 - (speed_lvl * 10)) # Рассчитывается пропорционально уровню понижения скорости
            
            else: 
                n_steps = 0

            shifted_sound = librosa.effects.pitch_shift(y=edited_sound_vector, sr=sample_rate, n_steps=n_steps)
            return shifted_sound, sample_rate
     

    def _plot_sound_wave(self, sample_rate: int, 
                               sound_vector: np.ndarray) -> None:
        """
        (СДЕЛАТЬ ДЛЯ УДОБСТВА. не относится к тз) Визуализация звукового сигнала .

        :param sample_rate: Частота дискретизации звукового файла.
        :param sound_vector: Массив данных звукового файла.
        """
        length = sound_vector.shape[0] / sample_rate
        time = np.linspace(0., length, sound_vector.shape[0])

        plt.figure(figsize=(10, 4))

        if sound_vector.ndim >= 2:
            plt.plot(time, sound_vector[:, 0], label='Left Channel')
            plt.plot(time, sound_vector[:, 1], label='Right Channel')

        else:
            plt.plot(time, sound_vector, label='Mono Channel')

        plt.legend()
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title('Sound Wave')
        plt.show()