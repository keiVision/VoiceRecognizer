import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np

class VoiceRecognizer:
    def __init__(self, model_path: str, *args):
        self.model_path = model_path
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model, self.processor = self.create_model()

    def create_model(self):
        model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_path, 
                                                          torch_dtype=self.torch_dtype,
                                                          low_cpu_mem_usage=True, 
                                                          use_safetensors=True)
        model.to(self.device)
        processor = AutoProcessor.from_pretrained(self.model_path)

        return model, processor

    def process_sound(self, sound_vector: np.ndarray, language: str = None):

        pipe = pipeline(
            task="automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
            generate_kwargs={"language": language}
            )
        result = pipe(sound_vector)
        
        return result['text']
    
