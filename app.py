import gradio as gr
import tensorrt_llm
import torch
import numpy as np
from run import decode_wav_file
from transformers import pipeline

device = 0 if torch.cuda.is_available() else "cpu"

fill_mask = pipeline(
    task="fill-mask",
    device=device,
)

sentiment = pipeline(
    task="sentiment-analysis",
    device=device,
)


def transcribe_live(stream, new_chunk):
    if new_chunk is None:
        raise gr.Error(
            "No audio file submitted! Please upload or record an audio file before submitting your request."
        )

    if stream is not None:
        stream = np.concatenate([stream, chunk_text])
    else:
        stream = chunk_text

    chunk_text = decode_wav_file(
        stream,
        return_duration_info=False,
    )

    return stream, chunk_text


def transcribe(inputs):
    if inputs is None:
        raise gr.Error(
            "No audio file submitted! Please upload or record an audio file before submitting your request."
        )

    text = decode_wav_file(inputs, return_duration_info=False)

    print(f"Original Text: {text}")  # print the original transcribed text

    # Masking the text if not already masked
    if "<mask>" in text:
        print("Found '<mask>' in text.")
        unmasked_text = fill_mask(text)[0]["sequence"]
        print(f"Unmasked Text: {unmasked_text}")  # print the text after unmasking
    elif "mask" in text:
        print("'mask' found in text, replacing with '<mask>' for processing.")
        replaced_text = text.replace("mask", "<mask>")
        unmasked_text = fill_mask(replaced_text)[0]["sequence"]
        print(
            f"Text after replacing 'mask' with '<mask>' and unmasking: {unmasked_text}"
        )
    else:
        unmasked_text = text
        print("No 'mask' or '<mask>' found in text.")

    # Custom Sentiment Mapping
    sentiment_result = sentiment(unmasked_text)
    sentiment_label = sentiment_result[0]["label"]
    sentiment_score = sentiment_result[0]["score"]

    # Map the sentiment to custom descriptions
    if sentiment_label == "POSITIVE" and sentiment_score > 0.85:
        custom_sentiment = "lowkey good vibes"
    elif sentiment_label == "POSITIVE":
        custom_sentiment = "kinda nice"
    elif sentiment_label == "NEGATIVE" and sentiment_score <= 0.85:
        custom_sentiment = "no strong feelings"
    else:
        custom_sentiment = "mixed vibes"

    # Combine the custom sentiment result into one string for both printing and returning
    sentiment_result_str = (
        f"Sentiment: {sentiment_label} with score {sentiment_score:.2f}"
    )
    print(sentiment_result_str)  # print custom sentiment analysis results

    return unmasked_text, custom_sentiment


if __name__ == "__main__":
    tensorrt_llm.logger.set_level("info")

    demo = gr.Blocks()

    mf_transcribe = gr.Interface(
        fn=transcribe,
        inputs=[
            gr.Audio(sources="microphone", type="filepath"),
        ],
        outputs=[
            gr.Textbox(label="Transcription"),
            gr.Textbox(label="Sentiment"),
        ],
        title="Pod-Helper Transcription Service",
        description=(
            "Transcribe long-form microphone or audio inputs with the click of a button! Demo uses the OpenAI Whisper"
        ),
        allow_flagging="never",
    )

    # live transcrption
    live_transcribe = gr.Interface(
        fn=transcribe_live,
        inputs=[
            "state",
            gr.Audio(sources="microphone", streaming=True),
        ],
        outputs=["state", gr.Textbox(label="Transcription")],
        live=True,
        title="Pod-Helper Transcription Service",
        description=("Streaming ASR Demo. Demo uses the OpenAI Whisper"),
        allow_flagging="never",
    )

    file_transcribe = gr.Interface(
        fn=transcribe,
        inputs=[
            gr.Audio(sources="upload", type="filepath"),
        ],
        outputs=[
            gr.Textbox(label="Transcription"),
            gr.Textbox(label="Sentiment"),
        ],
        title="Pod-Helper Transcription Service",
        description=(
            "Transcribe long-form microphone or audio inputs with the click of a button! Demo uses the OpenAI Whisper."
        ),
        allow_flagging="never",
    )

    with demo:
        gr.TabbedInterface(
            [mf_transcribe, live_transcribe, file_transcribe],
            ["Microphone", "Microphone Streaming", "Audio File upload"],
        )

    demo.launch()
