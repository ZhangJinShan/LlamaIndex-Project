import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from util import load_model

# 加载大模型和嵌入模型
llm = load_model.get_llm()
embed = load_model.get_embed()

# 加载文档
documents = SimpleDirectoryReader(input_files=["./data/小说.txt"]).load_data()
# 初始化节点解析器
splitter = SentenceSplitter(chunk_size=512)
nodes = splitter.get_nodes_from_documents(documents=documents)

# 创建文档存储器
doc_store = SimpleDocumentStore()
doc_store.add_documents(nodes)

# 创建chroma连接对象
client = chromadb.PersistentClient(path="./chroma_db_")
chroma_collection = client.get_or_create_collection(name="dense_vectors")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 创建上下文存储器
storage_context = StorageContext.from_defaults(docstore=doc_store, vector_store=vector_store)

# 创建向量索引
index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)

# 创建混合检索器
retriever = QueryFusionRetriever(
    retrievers=[
        index.as_retriever(similarity_top_k=2),
        BM25Retriever.from_defaults(
            docstore=doc_store,
            similarity_top_k=2
        )
    ],
    # 生成同义词的问题数量
    num_queries=1,
    use_async=True,
    verbose=True
)

nodes = retriever.retrieve("纳兰嫣然在哪个宗门修炼？")
for node in nodes:
    print(node)

print("===============" * 8)

query_engine = RetrieverQueryEngine(retriever=retriever)
print(query_engine.query("纳兰嫣然在哪个宗门修炼？"))


