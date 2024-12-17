import xml.etree.ElementTree as ET
import torch
from datetime import datetime
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import numpy as np
from enum import Enum
from typing import List, Dict, Union
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

class SegmentType(Enum):
    TEXT = "text"
    PAUSE = "pause"

def generate_meditation_ssml(topic: str) -> str:
    """
    Generate SSML meditation text using Claude API based on a given topic.
    
    Args:
        topic: str - The topic or focus of the meditation
    Returns:
        str - Generated SSML text
    """
    client = anthropic.Client(api_key=os.getenv('ANTHROPIC_API_KEY'))
    
    prompt = f"""Generate a gentle guided meditation SSML script about {topic}. 
    The meditation should:
    - Be in German
    - Be around 6-8 sentences long
    - Include appropriate pauses (using <break> tags) between sentences
    - Use calming, mindful language
    - Do not use abbrevations or anglicisms
    - Follow this SSML structure:
    
    <speak>
        <p>
            <s>[First sentence]</s>
            <break time="2s"/>
        </p>
        [Additional paragraphs...]
    </speak>
    
    Ensure each sentence is wrapped in <s> tags and followed by a break tag with appropriate duration (2-4 seconds).
    Make the pauses longer (3-4s) for more important moments of reflection.
    
    Make sure the output starts with <speak> and ends with </speak>.
    """

    message = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1000,
        temperature=0.7,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    return message.content[0].text

def ssml_to_json(ssml_string):
    """
    Convert SSML string to JSON format, considering only text and breaks.
    Breaks are always converted to seconds as integers.

    Args:
        ssml_string: SSML text as string
    Returns:
        List of dictionaries with text and pause information
    """
    try:
        # Parse SSML string
        root = ET.fromstring(ssml_string)
        
        def convert_to_seconds(time):
            if time.endswith("ms"):
                return int(int(time[:-2]) / 1000)
            elif time.endswith("s"):
                return int(float(time[:-1]))
            else:
                return 0
        
        def parse_element(element):
            result = []
            if element.text and element.text.strip():
                result.append({"type": "text", "content": element.text.strip()})
            
            if element.tag == "break" and "time" in element.attrib:
                pause_time = convert_to_seconds(element.attrib["time"])
                result.append({"type": "pause", "time": pause_time})
            
            for child in element:
                result.extend(parse_element(child))
            
            return result
        
        ssml_json = parse_element(root)
        return ssml_json
    
    except ET.ParseError as e:
        return {"error": f"SSML Parse Error: {e}"}

def text_to_speech(topic, segments: List[Dict[str, Union[str, float]]]):
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
    
    description = "Nicole's german voice is flat, seductive, slow, soft and meditative without any emotions. The recording is of very high quality, with the speaker's voice sounding clear and very close up."
    input_ids = description_tokenizer(description, return_tensors="pt").input_ids.to(device)
    
    audio_segments = []
    sampling_rate = model.config.sampling_rate
    
    for segment in segments:
        segment_type = SegmentType(segment["type"])
        
        if segment_type == SegmentType.TEXT:
            prompt_input_ids = tokenizer(segment["content"], return_tensors="pt").input_ids.to(device)
            generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
            audio_arr = generation.cpu().numpy().squeeze()
            audio_segments.append(audio_arr)
            
        elif segment_type == SegmentType.PAUSE:
            pause_samples = int(segment["time"] * sampling_rate)
            pause_arr = np.zeros(pause_samples)
            audio_segments.append(pause_arr)
    
    final_audio = np.concatenate(audio_segments)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"meditation_{timestamp}_{topic}.wav"
    sf.write(output_filename, final_audio, sampling_rate)
    
    return final_audio, output_filename

def generate_meditation(topic: str):
    """
    Generate and synthesize a meditation on a given topic.
    
    Args:
        topic: str - The topic or focus of the meditation
    Returns:
        tuple - (audio_array, output_filename)
    """
    # Generate SSML text using Claude
    ssml_text = generate_meditation_ssml(topic)
    
    # Convert SSML to JSON format
    meditation_segments = ssml_to_json(ssml_text)
    print(meditation_segments)
    
    # Generate audio
    # audio, filename = text_to_speech(topic, meditation_segments)
    
    # print(f"Meditation generated and saved as: {filename}")
    # return audio, filename

# python meditations.py "Anstrengende Programmierungs-Aufgaben werden gemeistert. Alle Probleme gelöst."
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate a meditation on a specific topic')
    parser.add_argument('topic', type=str, help='Topic or focus of the meditation')
    
    args = parser.parse_args()
    
    generate_meditation(args.topic)
