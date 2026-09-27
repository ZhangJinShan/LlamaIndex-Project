from pathlib import Path

from llama_index.core.node_parser import JSONNodeParser
from llama_index.readers.file import FlatReader

documents = FlatReader().load_data(Path("./data/request.json"))

nodes = JSONNodeParser().get_nodes_from_documents(documents=documents, show_progress=True)

print(len(nodes))
print(nodes)
