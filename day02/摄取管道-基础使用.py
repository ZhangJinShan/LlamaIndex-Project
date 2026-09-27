from llama_index.core import SimpleDirectoryReader
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter

from util.load_model import get_embed, get_llm

# 定义数据连接器去读取数据
reader = SimpleDirectoryReader(input_files=["./data/小说.txt"])
documents = reader.load_data()

# 初始化本地向量模型
embed = get_embed()
# 初始化大语言模型
llm = get_llm()

# 定义文本分割器
text_splitter = SentenceSplitter(chunk_size=256, chunk_overlap=30)

# 定义标题提取器
title_extractor = TitleExtractor(nodes=5, node_template="请为以下文档生成一个简洁的标题: {context_str}", num_workers=5)

# 创建数据摄入管道
pipeline = IngestionPipeline(transformations=[text_splitter, embed, title_extractor])

# 执行管道
nodes = pipeline.run(documents=documents)

for node in nodes:
    print(node, "\n\n")
    print(node.metadata, "\n\n")
    print("-------" * 10)


