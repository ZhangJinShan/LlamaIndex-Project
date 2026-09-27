import asyncio

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.agent import FunctionAgent

from util.load_model import get_embed, get_llm

# 关键点：Settings.embed_model / Settings.llm 必须在建索引【之前】设置好，
# 否则 VectorStoreIndex 会回落到默认的 OpenAI embedding 并报 No API key
llm = get_llm()
embed_model = get_embed()


# Define a simple calculator tool
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b

# Create a RAG tool using LlamaIndex
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
async def search_documents(query: str) -> str:
    """Useful for answering natural language questions about an personal essay written by Paul Graham."""
    response = await query_engine.aquery(query)
    return str(response)

agent = FunctionAgent(
    tools=[multiply, search_documents],
    llm=llm,
    system_prompt="You are a helpful assistant that can perform calculations \
    and search through documents to answer questions.",
    # 百炼(DashScope)集成在流式返回时会对未闭合的 tool call 参数做 json.loads 而报错，
    # 关闭流式走非流式分支即可正常解析工具调用
    streaming=False,
    verbose=True
)

async def main():
    response = await agent.run(
        "What did the author do in college? Also, what's 7 * 8?"
    )
    print(str(response))

if __name__ == "__main__":
    asyncio.run(main())
