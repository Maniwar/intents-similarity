import torch
import streamlit as st
import logging

logger = logging.getLogger(__name__)

PARAPHRASE_MODELS = {
    "humarin/chatgpt_paraphraser_on_T5_base": "ChatGPT-style (Fast, Creative)",
    "ramsrigouthamg/t5-large-paraphraser-diverse-high-quality": "T5-Large (High Quality, Slower)",
    "Vamsi/T5_Paraphrase_Paws": "PAWS (Original, May Have Issues)",
}


@st.cache_resource
def load_paraphrase_pipeline(model_name, use_gpu):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

    logger.info("Loading paraphrase model: %s (GPU=%s)", model_name, use_gpu)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = 0 if use_gpu else -1
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)


def generate_paraphrases(
    phrases,
    model_name,
    use_gpu=False,
    num_variations=2,
    max_phrases=20,
    temperature=0.7,
):
    generator = load_paraphrase_pipeline(model_name, use_gpu)

    sample = phrases[:max_phrases]
    new_phrases = []
    for phrase in sample:
        for _ in range(num_variations):
            if "chatgpt_paraphraser" in model_name:
                input_text = f"paraphrase: {phrase}"
            else:
                input_text = f"paraphrase: {phrase} </s>"

            result = generator(
                input_text,
                max_new_tokens=128,
                num_return_sequences=1,
                do_sample=True,
                temperature=temperature,
            )
            text = result[0]['generated_text'].replace("paraphrased: ", "").strip()
            new_phrases.append(text)

    return new_phrases, len(phrases) > max_phrases
