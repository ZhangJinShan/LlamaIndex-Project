from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.vector_stores import SimpleVectorStore

from util.load_model import get_llm, get_embed

# 加载 llm 和 embedding 模型
llm = get_llm()
embed = get_embed()

# 加载文档并构建索引
documents = SimpleDirectoryReader(input_files=["./data/小说.txt"]).load_data()

print("-------------------使用最基础的内存向量进行本地存储----------------------")
# 创建一个最基础的内存向量
simple_vector_store = SimpleVectorStore()
# 创建一个存储容器
storage_context = StorageContext.from_defaults(vector_store=simple_vector_store)
# 创建索引
index = VectorStoreIndex.from_documents(documents=documents, storage_context=storage_context, show_progress=True)
# 查询数据
query_engine = index.as_query_engine()
response = query_engine.query("古河是谁？")
print(response)
# 将数据存储到本地
storage_context.persist("./storage")

# 从本地加载已存储的向量数据
storage_context_new = StorageContext.from_defaults(persist_dir="./storage")
# 通过load_index_from_storage去加载本地保存的index
new_index = load_index_from_storage(storage_context_new)
new_response = new_index.as_query_engine().query("谁要和萧炎退婚")
print(new_response)