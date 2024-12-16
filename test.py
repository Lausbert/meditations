from transformers import pipeline
from playsound import playsound
import os
import tempfile
 
def text_to_speech(text, model_name="espnet/kan-bayashi-ljspeech-vits"):
    """
    Wandelt Text in Sprache um und gibt die Ausgabe als Audiodatei wieder.
 
    :param text: Der Text, der in Sprache umgewandelt werden soll
    :param model_name: Der Name des TTS-Modells auf Hugging Face (Standard: LJSpeech VITS-Modell)
    """
    try:
        # Text-to-Speech-Pipeline laden
        tts = pipeline("text-to-speech", model=model_name)
 
        # Text in Sprache umwandeln
        print("Wandle Text in Sprache um...")
        output = tts(text)
 
        # Temporäre Datei speichern
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(output[0]["array"])
            audio_path = temp_audio.name
 
        # Audiodatei abspielen
        print("Spiele Audiodatei ab...")
        playsound(audio_path)
 
        # Temporäre Datei löschen
        os.remove(audio_path)
 
    except Exception as e:
        print(f"Fehler bei der Text-zu-Sprache-Konvertierung: {e}")
 
if __name__ == "__main__":
    # Beispieltext
    text = "Hallo, wie geht es dir heute? Ich bin ein Text-zu-Sprache-Modell."
 
    # Funktion aufrufen
    text_to_speech(text)