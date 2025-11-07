import logging
import os

import whisper
import ollama
# from triton.knobs import language

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# def optimize_transcription_result(text):
#     prompt = f"""
#        Aşağıdaki transkripti Türkçe dilinde anlamı bozulmadan düzelt.
#        Gerektiğinde eksik kelimeleri bağlamdan tamamla:
#        "{text}"
#        """
#     response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
#     return response['message']['content']


def transcribe_audio(transcription_file):
    log.info("Transcribing audio...")
    transcribed = whisper.load_model("large-v3-turbo", device="cpu").transcribe(transcription_file, language="tr", fp16=False)
    log.info(f"Transcription completed\nResult:\n{transcribed}")
    return transcribed
    # return optimize_transcription_result(transcribed)


def process_audio(contents):
    file_name = create_temporary_file(contents)
    return transcribe_audio(file_name)


def create_temporary_file(contents):
    with open("audio.mp3", "wb") as audio_file:
        audio_file.write(contents)
        audio_file.close()
        return os.path.abspath(audio_file.name)
