from llama_index.core import SimpleDirectoryReader
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor, SummaryExtractor, KeywordExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter

from util.load_model import get_llm, get_embed

llm = get_llm()
embed_model = get_embed()

# 定义数据连接器去读取数据
documents = SimpleDirectoryReader(input_files=["./data/小说.txt"]).load_data()

# 创建管道中转换组件
transformations = [
    SentenceSplitter(),
    TitleExtractor(nodes=5),
    QuestionsAnsweredExtractor(questions=3),
    SummaryExtractor(summaries=["prev", "self"]),
    KeywordExtractor(keywords=10)
]
# 创建摄取管道
pipeline = IngestionPipeline(transformations=transformations)

nodes = pipeline.run(documents=documents, show_progress=True)

print(nodes)