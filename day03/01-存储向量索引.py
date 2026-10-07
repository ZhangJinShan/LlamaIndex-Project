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

# 定义本地向量数据库
client = chromadb.PersistentClient(path="./chroma_db") # 定义一个库
collection = client.get_or_create_collection(name="quick_start") # 定义一个表
vector_store = ChromaVectorStore(chroma_collection=collection)

# 构建向量存储并自定义存储上下文
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 加载文档并构建索引
documents = SimpleDirectoryReader(input_files=["./data/deepseek介绍.txt"]).load_data()

# 方式1：从文档构建索引
# vector_store_index = VectorStoreIndex.from_documents(documents=documents, storage_context=storage_context, show_progress=True)

# 方式2：从节点构建索引
# 构建数据处理摄取管道
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=30),  # 分割成512字符的句子，重叠30个字符
        TitleExtractor(), # 提取标题
        embed # 嵌入向量
    ],
    vector_store=vector_store)
nodes = pipeline.run(documents=documents, show_progress=True) # 切分成节点

# 构建索引
vector_store_index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, show_progress=True)

# 索引查询
response = vector_store_index.as_retriever().retrieve("deepseek介绍")

print(response)
