import chromadb
from llama_index.core import StorageContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from util.load_model import get_llm, get_embed

# 加载 llm 和 embedding 模型
llm = get_llm()
embed = get_embed()

# 加载文档并构建索引
documents = SimpleDirectoryReader(input_files=["./data/小说.txt"]).load_data()

print("---------------使用chroma进行存储向量--------------------")
# 定义本地向量数据库
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="quick_start")
vector_store = ChromaVectorStore(chroma_collection=collection)

# 构建向量存储并自定义存储上下文
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 方式1：从文档构建索引
# vector_store_index = VectorStoreIndex.from_documents(documents=documents, storage_context=storage_context, show_progress=True)

# 方式2：从节点构建索引
pipeline = IngestionPipeline(
    transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=30), TitleExtractor(), embed],
    vector_store=vector_store)
nodes = pipeline.run(documents=documents, show_progress=True)
vector_store_index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, show_progress=True)

print(collection.count())
query_engine = vector_store_index.as_query_engine()
response = query_engine.query("古河是谁？")
print(response)

print("---------------使用chroma获取存储向量--------------------")
chroma_collection_new = client.get_collection("quick_start")
vector_store_new = ChromaVectorStore(chroma_collection=chroma_collection_new)
# 加载索引（只恢复索引结构，不重新写入）
index_new = VectorStoreIndex.from_vector_store(
    vector_store=vector_store_new,
    embed_model=embed  # 必须与原来用的一致
)

# 可以开始查询
query_engine_new = index_new.as_query_engine()
response = query_engine_new.query("谁要和萧炎退婚？")
print(response)
