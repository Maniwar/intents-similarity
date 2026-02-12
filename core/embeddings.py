import numpy as np
import torch
import streamlit as st
from contextlib import nullcontext
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import logging

logger = logging.getLogger(__name__)

SENTENCE_TRANSFORMER_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "label": "all-MiniLM-L6-v2",
        "icon": "⚡",
        "size": "80MB",
        "description": "Ultra-fast & lightweight | 14k sentences/sec | 85% STS-B | English only",
    },
    "sentence-transformers/all-mpnet-base-v2": {
        "label": "all-mpnet-base-v2",
        "icon": "🏆",
        "size": "420MB",
        "description": "Best quality-speed balance | 4k sentences/sec | 88% STS-B | English only",
    },
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": {
        "label": "paraphrase-multilingual-mpnet-base-v2",
        "icon": "🌍",
        "size": "1.1GB",
        "description": "Multilingual champion | 50+ languages | Excellent cross-lingual quality | Default",
    },
    "BAAI/bge-base-en-v1.5": {
        "label": "bge-base-en-v1.5",
        "icon": "🚀",
        "size": "440MB",
        "description": "SOTA English | Top MTEB scores | Instruction-aware embeddings",
    },
    "BAAI/bge-m3": {
        "label": "bge-m3",
        "icon": "🌐",
        "size": "2.3GB",
        "description": "Multi-everything | 100+ languages | Multi-granularity | Slower but powerful",
    },
    "intfloat/e5-base-v2": {
        "label": "e5-base-v2",
        "icon": "⚖️",
        "size": "440MB",
        "description": "Balanced performer | Good speed-quality tradeoff | English only",
    },
    "intfloat/multilingual-e5-base": {
        "label": "multilingual-e5-base",
        "icon": "🌏",
        "size": "1.1GB",
        "description": "E5 multilingual | 100+ languages | Good MIRACL scores",
    },
    "nomic-ai/nomic-embed-text-v1.5": {
        "label": "nomic-embed-text-v1.5",
        "icon": "🔬",
        "size": "550MB",
        "description": "Research leader | Top BEIR scores | Multimodal-ready",
    },
    "Alibaba-NLP/gte-base-en-v1.5": {
        "label": "gte-base-en-v1.5",
        "icon": "🎯",
        "size": "440MB",
        "description": "Precision model | Angle-optimized | Strong retrieval",
    },
    "sentence-transformers/xlm-r-large": {
        "label": "xlm-r-large",
        "icon": "🔧",
        "size": "2.2GB",
        "description": "Legacy large | XLM-RoBERTa base | 100 languages",
    },
    "sentence-transformers/xlm-r-bert-base-nli-stsb-mean-tokens": {
        "label": "xlm-r-bert-base-nli-stsb-mean-tokens",
        "icon": "📊",
        "size": "1.1GB",
        "description": "NLI-trained | Trained on STS benchmarks",
    },
    "sentence-transformers/LaBSE": {
        "label": "LaBSE",
        "icon": "🔄",
        "size": "1.8GB",
        "description": "Universal encoder | 109 languages | Cross-lingual search",
    },
}

BASE_TRANSFORMER_MODELS = [
    "xlm-roberta-large",
    "xlm-roberta-base",
    "FacebookAI/xlm-roberta-large",
    "FacebookAI/xlm-roberta-base",
]


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


@st.cache_resource
def load_sentence_transformer(model_name, device_str):
    logger.info("Loading sentence transformer: %s on %s", model_name, device_str)
    model = SentenceTransformer(model_name)
    model = model.to(device_str)
    return model


@st.cache_resource
def load_base_transformer(model_name, device_str):
    logger.info("Loading base transformer: %s on %s", model_name, device_str)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
    device = torch.device(device_str)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        if param.device != device:
            param.data = param.data.to(device)
    return model, tokenizer


def encode_texts(
    texts,
    model,
    tokenizer=None,
    model_type='sentence_transformer',
    device_obj=None,
    batch_size=32,
    normalize_embeddings=True,
    use_mixed_precision=True,
    show_progress=False,
    pooling_strategy='mean',
):
    if device_obj is None:
        device_obj = torch.device('cpu')
    batch_size = max(int(batch_size), 1)

    if model_type == 'sentence_transformer':
        device_str = str(device_obj)
        if device_obj.type != 'cpu':
            model = model.to(device_obj)
        encode_kwargs = {
            "device": device_str,
            "batch_size": batch_size,
            "convert_to_numpy": True,
            "show_progress_bar": show_progress and len(texts) > batch_size,
        }
        if normalize_embeddings:
            encode_kwargs["normalize_embeddings"] = True
        try:
            return model.encode(texts, **encode_kwargs)
        except TypeError:
            encode_kwargs.pop("normalize_embeddings", None)
            return model.encode(texts, **encode_kwargs)
    else:
        model_device = next(model.parameters()).device
        if model_device != device_obj:
            model.to(device_obj)

        embeddings_list = []
        autocast_enabled = use_mixed_precision and device_obj.type == 'cuda'
        autocast_ctx = torch.cuda.amp.autocast if autocast_enabled else nullcontext

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded_input = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512,
            )
            encoded_input = {k: v.to(device_obj) for k, v in encoded_input.items()}

            with torch.no_grad():
                with autocast_ctx():
                    model_output = model(**encoded_input)

            token_embeddings = (
                model_output.last_hidden_state
                if hasattr(model_output, "last_hidden_state")
                else model_output[0]
            )

            if pooling_strategy == 'cls':
                batch_embeddings = token_embeddings[:, 0, :]
            elif pooling_strategy == 'mean':
                batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            else:
                raise ValueError(f"Unsupported pooling strategy: {pooling_strategy}")

            if normalize_embeddings:
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            embeddings_list.append(batch_embeddings.detach().cpu())

        if not embeddings_list:
            return np.array([])
        return torch.cat(embeddings_list, dim=0).numpy()
