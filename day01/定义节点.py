from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

# 加载文件
documents = SimpleDirectoryReader("./data").load_data()

# 进行切片
parser = SentenceSplitter()
# 将文档解析成节点
nodes = parser.get_nodes_from_documents(documents)
for node in nodes:
    print(node)
    print("----------" * 10)