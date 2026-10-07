from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.dashscope import DashScope
from dotenv import load_dotenv
import os

from llama_index.llms.deepseek import DeepSeek

load_dotenv()


def get_llm(model: str = "qwen-plus"):
    api_key = os.getenv("DASHSCOPE_API_KEY")
    api_base_url = os.getenv("DASHSCOPE_BASE_URL")

    # LlamaIndex默认使用的大模型被替换为百炼
    llm = DashScope(model_name=model, api_key=api_key, api_base=api_base_url, is_chat_model=True)
    Settings.llm = llm

    return llm


def get_embed():
    # 加载本地的嵌入模型
    model_name = r"/Users/zhangjinshan/LLM/local_model/BAAI/bge-small-zh-v1.5"
    # model_name = r"/Users/zhangjinshan/LLM/local_model/BAAI/bge-large-zh-v1___5"
    embed_model = HuggingFaceEmbedding(model_name=model_name, device="cpu", max_length=512)
    # 设置默认的向量模型为本地模型（不设置会回落到 OpenAI embedding 并报 No API key）
    Settings.embed_model = embed_model

    return embed_model


def get_deepseek_llm(model: str = "deepseek-flash"):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    api_base_url = os.getenv("DEEPSEEK_BASE_URL")
    # LlamaIndex默认使用的大模型被替换为百炼
    llm = DeepSeek(model=model, api_key=api_key, api_base=api_base_url, is_chat_model=True)
    Settings.llm = llm

    return llm

