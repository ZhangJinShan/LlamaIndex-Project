from typing import List

from llama_index.core import SimpleDirectoryReader, QueryBundle, StorageContext, VectorStoreIndex, \
    SimpleKeywordTableIndex, get_response_synthesizer
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.keyword_table import KeywordTableSimpleRetriever
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.schema import NodeWithScore

from util import load_model

# 加载大模型和嵌入模型
llm = load_model.get_llm()
embed = load_model.get_embed()

# 加载文档
documents = SimpleDirectoryReader(input_files=["./data/小说.txt"]).load_data()
# 初始化节点解析器
splitter = SentenceSplitter(chunk_size=512)
nodes = splitter.get_nodes_from_documents(documents)

class CustomRetriever(BaseRetriever):
    """
    执行语义搜索和简单关键字搜索的自定义检索器。
    """
    def __init__(
            self,
            # 向量检索器
            vector_retriever: VectorIndexRetriever,
            # 关键词检索器
            keyword_retriever: KeywordTableSimpleRetriever,
            mode: str = "AND"
    ) -> None:
        """Init params"""
        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        if mode not in ["AND", "OR"]:
            raise ValueError(f"mode must be one of 'AND' or 'OR', but got {mode}")
        self._mode = mode
        super().__init__()


    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        vector_ids = {
            n.node.node_id for n in vector_nodes
        }
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)
        keyword_ids = {
            n.node.node_id for n in keyword_nodes
        }
        combined_dict = {
            n.node.node_id: n for n in vector_nodes
        }
        combined_dict.update({
            n.node.node_id: n for n in keyword_nodes
        })

        if self._mode == "AND":
            # 获取两个检索器交集的数据， 获取两个检索器中有着共同node_id的数据
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            # 获取两个检索器并集的数据， 获取两个检索器中有着共同node_id的数据
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]

        return retrieve_nodes

# 初始化上下文存储器
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

# 创建对应的索引
vector_index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
# 简单关键词索引，适合结构化数据或者短文本查询
keyword_index = SimpleKeywordTableIndex(nodes=nodes, storage_context=storage_context)

# 定义自定义检索器
vector_index_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=2)
keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index)
custom_retriever = CustomRetriever(
    vector_retriever=vector_index_retriever,
    keyword_retriever=keyword_retriever,
    mode="AND"
)

# 定义响应合成器
response_synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.TREE_SUMMARIZE,
    streaming=True
)

# 加载查询引擎
custom_query_engine = RetrieverQueryEngine(
    retriever=custom_retriever,
    response_synthesizer=response_synthesizer
)

response = custom_query_engine.query("斗之气：九段！级别：高级！是谁")
print(response)

