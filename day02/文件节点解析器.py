from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.readers.file import FlatReader
from pathlib import Path

# 读取文件 FlatReader：从文件中提取原始文本
md_docs = FlatReader().load_data(Path("./data/小说.txt"))

# 创建节点解析器，根据后缀名选择对应的解析器
parser = SimpleFileNodeParser()
# 将文档解析成节点
nodes = parser.get_nodes_from_documents(documents=md_docs, show_progress=True)
print(len(nodes))
print(nodes)
