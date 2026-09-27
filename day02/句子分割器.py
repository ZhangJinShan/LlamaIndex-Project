from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size=512,  # 分割长度
    chunk_overlap=50,  # 重叠长度
    paragraph_separator="\r\n\r\n",  # 段落分割符 第一优先级
    secondary_chunking_regex="[^，。；！？]+[，。；！？]?"  # 二级切分正则表达式 第二优先级
)

documents = SimpleDirectoryReader(input_files=["./data/小说.txt"]).load_data()

nodes = splitter.get_nodes_from_documents(documents=documents, show_progress=True)

for node in nodes:
    print(node.text, "---"*10)
