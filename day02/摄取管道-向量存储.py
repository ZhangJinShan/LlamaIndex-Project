import time

import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from util.load_model import get_llm, get_embed

llm = get_llm()
embed = get_embed()

# 定义数据连接器去读取数据
reader = SimpleDirectoryReader(input_files=["./data/小说.txt"])
documents = reader.load_data()

# 定义本地向量数据库
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.get_or_create_collection(name="quick_start")

# 创建 Chroma 向量数据库对象
vector_store = ChromaVectorStore(chroma_collection=chroma_collection, embedding_function=embed)

# 定义文档分割器
text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=30)

# 创建摄取管道
pipeline = IngestionPipeline(transformations=[text_splitter, embed], vector_store=vector_store)

# 开始时间
start = time.time()

# 执行管道
pipeline.run(documents=documents, show_progress=True)
# 统计文档加载的时间
time2 = time.time() - start
print(f">>> 第一次处理文档，耗时: {time2:.2f}秒")

# 将管道持久化到本地
pipeline.persist(persist_dir="./pipeline_storage")

# 重新加载管道
# 加载和恢复状态
new_pipeline = IngestionPipeline(
    transformations=[text_splitter, embed], vector_store=vector_store
)
# 从缓存中读取持久化管道数据
new_pipeline.load(persist_dir="./pipeline_storage")

# 开始时间
new_start = time.time()
# 由于缓存的存在会立即执行
nodes = new_pipeline.run(documents=documents, show_progress=True)
# 统计文档加载的时间
new_time2 = time.time() - new_start
print(f">>> 第二次处理文档，耗时: {new_time2:.2f}秒")

# # 打印处理后的节点
# for node in nodes:
#     print(node, "-------", "\n\n")


# 创建索引对象
index = VectorStoreIndex.from_vector_store(vector_store)
# 创建检索器
retriever = index.as_retriever(similarity_top_k=3)
print(retriever.retrieve("萧薰儿的斗之气是多少？"))
