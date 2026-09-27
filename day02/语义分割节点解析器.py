from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser

from util.load_model import get_embed

documents = SimpleDirectoryReader(input_files=["./data/小说.txt"]).load_data()

embed = get_embed()

parser = SemanticSplitterNodeParser(embed_model=embed, buffer_size=1, breakpoint_percentile_threshold=95)

nodes = parser.get_nodes_from_documents(documents=documents, show_progress=True)

# 打印生成的节点
for node in nodes:
    print(node.text, node.metadata, "------")
