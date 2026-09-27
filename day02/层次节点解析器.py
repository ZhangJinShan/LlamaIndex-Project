from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore

from util.load_model import get_embed

# 读取数据
documents = SimpleDirectoryReader(input_files=['./data/小说.txt']).load_data()

node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 1024, 512])

# 文档转换成节点
nodes = node_parser.get_nodes_from_documents(documents)
for node in nodes:
    print(f"ID: {node.node_id}, Text: {node.text}...")
    if node.parent_node:
        print(f"Parent: {node.parent_node.node_id}")

# 获取所有的叶子节点
leaf_nodes = get_leaf_nodes(nodes)

get_embed()

# 创建文档存储
document_store = SimpleDocumentStore()
# 添加文档
document_store.add_documents(nodes)
# 创建需要存储的上下文
storage_context = StorageContext.from_defaults(docstore=document_store)

# 构建基础向量检索索引：仅对叶节点构建
base_index = VectorStoreIndex(nodes=leaf_nodes, storage_context=storage_context)

base_retriever = base_index.as_retriever(similarity_top_k=6)

retriever = AutoMergingRetriever(
    vector_retriever=base_retriever,
    storage_context=storage_context,
    simple_ratio_thresh=0.5,  # 控制合并阈值
    verbose=True  # 显示合并日志
)

# 6. 查询
query_str = "萧炎现在是什么段位？"
nodes_returned = retriever.retrieve(query_str)
print(f"Retrieved {len(nodes_returned)} nodes:")
for node in nodes_returned:
    print("---")
    print(node.get_content())

base_nodes_returned = base_retriever.retrieve(query_str)
print(f"Retrieved {len(base_nodes_returned)} nodes:")
for node in base_nodes_returned:
    print("---")
    print(node.get_content())

