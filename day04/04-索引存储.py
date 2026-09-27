from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage

from util.load_model import get_llm, get_embed

# 加载 llm 和 embedding 模型
llm = get_llm()
embed = get_embed()

# 加载文档
documents = SimpleDirectoryReader(input_files=["./data/小说.txt"]).load_data()

# 创建索引
index = VectorStoreIndex.from_documents(documents)
print(index.as_query_engine().query("古河是谁"))
print(index.as_query_engine().query("谁要和萧炎退婚？"))

# 将索引存储在本地
index.storage_context.persist("./vector_store_index")

print("-----------------从本地加载已存储的索引数据------------------")

# 从本地加载已存储的索引数据
new_storage_context = StorageContext.from_defaults(persist_dir="./vector_store_index")
new_index = load_index_from_storage(new_storage_context)
print(new_index.as_query_engine().query("古河是谁"))
print(new_index.as_query_engine().query("谁要和萧炎退婚"))