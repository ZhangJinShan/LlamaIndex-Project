from llama_index.core import SimpleDirectoryReader
from llama_index.core.indices.property_graph import ImplicitPathExtractor
from llama_index.core.node_parser import SentenceSplitter

# 加载文档
documents = SimpleDirectoryReader(
    input_files=["./data/小说.txt"]
).load_data()

splitter = SentenceSplitter()
nodes = splitter.get_nodes_from_documents(documents)


kg_extractor = ImplicitPathExtractor()

extracted_nodes = kg_extractor(nodes=nodes, show_progress=True)
for node in extracted_nodes:
    # print("节点文本：", node.text)
    # print("提取的关系：", node.metadata.get("relations", []))
    print(node.dict())
    print("----------" * 8)