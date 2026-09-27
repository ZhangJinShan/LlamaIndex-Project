from pathlib import Path

from llama_index.core.node_parser import HTMLNodeParser
from llama_index.readers.file import FlatReader

documents = FlatReader().load_data(Path("./data/index.html"))

nodes = HTMLNodeParser(tags=["p", "h1", "li"]).get_nodes_from_documents(documents=documents, show_progress=True)

print(len(nodes))
print(nodes)
