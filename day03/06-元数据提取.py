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
    SentenceSplitter(), # 分割句子
    TitleExtractor(nodes=5), # 提取标题
    QuestionsAnsweredExtractor(questions=3), # 提取问题
    SummaryExtractor(summaries=["prev", "self"]), # 提取摘要
    KeywordExtractor(keywords=10) # 提取关键词
]
# 其实上面这几个提取器就是按照对应的规则将内容提取后统一放到了 metadata 中
# 所以叫元数据提取器

# 创建摄取管道
pipeline = IngestionPipeline(transformations=transformations)

nodes = pipeline.run(documents=documents, show_progress=True)

for node in nodes:
    print(node)
    print("*****" * 10)
    print(node.metadata)
    print("-----" * 10)