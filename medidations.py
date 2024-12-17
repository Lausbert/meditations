import xml.etree.ElementTree as ET
import torch
from datetime import datetime
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import numpy as np
from enum import Enum
from typing import List, Dict, Union

def ssml_to_json(ssml_string):
    """
    Konvertiert eine SSML-Zeichenkette in JSON-Format, wobei nur Texte und Pausen (break) berücksichtigt werden.
    Pausen werden immer in Sekunden als Integer angegeben.

    :param ssml_string: SSML-Text als String
    :return: JSON-Format als Dictionary
    """
    try:
        # Parsen des SSML-Strings
        root = ET.fromstring(ssml_string)
        
        # Hilfsfunktion zur Konvertierung der Pause in Sekunden
        def convert_to_seconds(time):
            if time.endswith("ms"):
                return int(int(time[:-2]) / 1000)
            elif time.endswith("s"):
                return int(float(time[:-1]))
            else:
                return 0  # Defaultwert, falls unbekanntes Format
        
        # Funktion zum Rekursiven Traversieren der SSML-Baumstruktur
        def parse_element(element):
            result = []
            # Text verarbeiten
            if element.text and element.text.strip():
                result.append({"type": "text", "content": element.text.strip()})
            
            # Break-Elemente verarbeiten
            if element.tag == "break" and "time" in element.attrib:
                pause_time = convert_to_seconds(element.attrib["time"])
                result.append({"type": "pause", "time": pause_time})
            
            # Rekursiv die Kinder verarbeiten
            for child in element:
                result.extend(parse_element(child))
            
            return result
        
        # Konvertierung starten
        ssml_json = parse_element(root)
        return ssml_json
    
    except ET.ParseError as e:
        return {"error": f"SSML Parse Error: {e}"}

class SegmentType(Enum):
    TEXT = "text"
    PAUSE = "pause"

def text_to_speech(segments: List[Dict[str, Union[str, float]]]):
    """
    Convert an array of segment objects to speech with pauses.
    
    Args:
        segments: List of dictionaries with either:
            - {"type": "text", "content": str} for text segments
            - {"type": "pause", "time": float} for pause segments
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-multilingual-v1.1").to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-multilingual-v1.1")
    description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    
    description = "Nicole's german voice is flat, seductive, slow, soft and meditative without any emotions. Make a breath after each sentence. Use a very close recording that almost has no background noise."
    input_ids = description_tokenizer(description, return_tensors="pt").input_ids.to(device)
    
    audio_segments = []
    sampling_rate = model.config.sampling_rate
    
    for segment in segments:
        segment_type = SegmentType(segment["type"])
        
        if segment_type == SegmentType.TEXT:
            # Generate speech for text content
            prompt_input_ids = tokenizer(segment["content"], return_tensors="pt").input_ids.to(device)
            generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
            audio_arr = generation.cpu().numpy().squeeze()
            audio_segments.append(audio_arr)
            
        elif segment_type == SegmentType.PAUSE:
            # Create silence for specified duration
            pause_samples = int(segment["time"] * sampling_rate)
            pause_arr = np.zeros(pause_samples)
            audio_segments.append(pause_arr)
    
    # Concatenate all segments
    final_audio = np.concatenate(audio_segments)
    
    # Save the result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sf.write(f"parler_tts_out-{timestamp}.wav", final_audio, sampling_rate)
    
    return final_audio

if __name__ == "__main__":
    ssml_input = """
<speak>
    <p>
        <s>Schließen Sie sanft Ihre Augen.</s>
        <break time="2s"/>
    </p>
    
    <p>
        <s>Spüren Sie, wie Ihr Körper von der Unterlage getragen wird.</s>
        <break time="3s"/>
    </p>
    
    <p>
        <s>Lassen Sie alle Anspannung los.</s>
        <break time="2.5s"/>
    </p>
    
    <p>
        <s>Richten Sie nun Ihre Aufmerksamkeit auf Ihren natürlichen Atem.</s>
        <break time="3s"/>
    </p>
    
    <p>
        <s>Beobachten Sie, wie die Luft durch Ihre Nase ein- und ausströmt.</s>
        <break time="3s"/>
    </p>
    
    <p>
        <s>Sie müssen nichts verändern. Beobachten Sie einfach.</s>
        <break time="2s"/>
    </p>
</speak>
    """

    # Konvertierung SSML -> JSON
    meditation_segments = ssml_to_json(ssml_input)

    # Generate the audio
    text_to_speech(meditation_segments)
