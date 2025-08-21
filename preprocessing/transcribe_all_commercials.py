import os
import glob
import json
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import warnings
from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=200,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

for audio_fn in tqdm(glob.glob('files/trimmed_audio/*.mp3')):

    stimulus_id = os.path.basename(audio_fn).split(".")[0]

    # catch warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") 
        transcription = pipe(audio_fn, generate_kwargs={"language": "english"})["text"]

    # print transcription
    #print(transcription)

    # save transcription
    with open(f"files/transcripts/{stimulus_id}.txt", "w") as f:
        f.write(transcription.encode('ascii', 'ignore').decode('ascii')) # cast to ascii to avoid unicode errors