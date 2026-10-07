from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore

# 加载文档并构建索引
documents = SimpleDirectoryReader(input_files=["./data/小说.txt"]).load_data()

# 解析成节点
nodes = SentenceSplitter().get_nodes_from_documents(documents)

print("-------------------使用最基础的内存文档进行本地存储----------------------")
# 创建简单文档存储，并把节点传入
doc_store = SimpleDocumentStore()
doc_store.add_documents(nodes)

# 创建一个存储容器
storage_context = StorageContext.from_defaults(docstore=doc_store)

# 将文件进行本地存储
storage_context.persist("./documents")

# 从本地加载已存储的向量数据
new_storage_context = StorageContext.from_defaults(persist_dir="./documents")
print(new_storage_context.docstore.docs)

print("-------------------使用Redis进行存储文档----------------------")
pass