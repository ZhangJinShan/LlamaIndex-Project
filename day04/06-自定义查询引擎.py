from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, PromptTemplate, get_response_synthesizer
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.query_engine.custom import STR_OR_RESPONSE_TYPE
from llama_index.core.response_synthesizers import BaseSynthesizer, ResponseMode
from llama_index.llms.dashscope import DashScope

from util import load_model

# 加载大模型和嵌入模型
llm = load_model.get_llm()
embed = load_model.get_embed()

# 加载文档
documents = SimpleDirectoryReader(input_files=["./data/小说.txt"]).load_data()

# 创建索引和检索器
index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever()

# 创建提示词模板
qa_prompt = PromptTemplate(
    "下面是上下文信息\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "请根据给定的上下文来回答问题 "
    "请回答这个问题\n"
    "Query: {query_str}\n"
    "Answer: "
)

class RAGStringQueryEngine(CustomQueryEngine):
    """RAG字符串查询引擎"""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer
    llm: DashScope
    qa_prompt: PromptTemplate

    def custom_query(self, query_str: str) -> STR_OR_RESPONSE_TYPE:
        nodes = self.retriever.retrieve(query_str)

        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        print("查询到的上下文->", context_str)
        response = self.llm.complete(
            qa_prompt.format(context_str=context_str, query_str=query_str)
        )
        return str(response)


# 配置响应合成器
synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.TREE_SUMMARIZE,
    streaming=True
)

# 使用自定义查询引擎
query_engine = RAGStringQueryEngine(
    retriever=retriever,
    response_synthesizer=synthesizer,
    llm=llm,
    qa_prompt=qa_prompt,
)

res = query_engine.query("萧炎的戒指是谁送给他的？")
print(res)
