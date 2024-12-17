import torch
from datetime import datetime
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import numpy as np

def text_to_speech(text_segments):
    """
    Convert an array of text segments to speech with configurable pauses between them.
    
    Args:
        text_segments: List of dictionaries with 'text' and 'pause' properties
            text (str): The text to convert to speech
            pause (float): The pause duration in seconds after this segment
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-multilingual-v1.1").to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-multilingual-v1.1")
    description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    
    description = "Nicole's german voice is flat, seductive, slow, soft and meditative without any emotions. Make a breath pause of three seconds between each sentence. Use a very close recording that almost has no background noise."
    input_ids = description_tokenizer(description, return_tensors="pt").input_ids.to(device)
    
    # List to store all audio segments
    audio_segments = []
    sampling_rate = model.config.sampling_rate
    
    # Process each text segment
    for segment in text_segments:
        # Generate speech for text
        prompt_input_ids = tokenizer(segment['text'], return_tensors="pt").input_ids.to(device)
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()
        audio_segments.append(audio_arr)
        
        # Add pause if specified
        if segment['pause'] > 0:
            pause_samples = int(segment['pause'] * sampling_rate)
            pause_arr = np.zeros(pause_samples)
            audio_segments.append(pause_arr)
    
    # Concatenate all segments
    final_audio = np.concatenate(audio_segments)
    
    # Save the result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sf.write(f"parler_tts_out-{timestamp}.wav", final_audio, sampling_rate)
    
    return final_audio

if __name__ == "__main__":
    # Example usage with text segments and pauses
    meditation_segments = [
        {
            "text": "Schließen Sie sanft Ihre Augen.",
            "pause": 2.0
        },
        {
            "text": "Spüren Sie, wie Ihr Körper von der Unterlage getragen wird.",
            "pause": 3.0
        },
        {
            "text": "Lassen Sie alle Anspannung los.",
            "pause": 2.5
        },
        {
            "text": "Richten Sie nun Ihre Aufmerksamkeit auf Ihren natürlichen Atem.",
            "pause": 3.0
        },
        {
            "text": "Beobachten Sie, wie die Luft durch Ihre Nase ein- und ausströmt.",
            "pause": 3.0
        },
        {
            "text": "Sie müssen nichts verändern. Beobachten Sie einfach.",
            "pause": 2.0
        }
    ]
    
    # Generate the audio
    text_to_speech(meditation_segments)
