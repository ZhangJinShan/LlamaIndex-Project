import asyncio

from llama_index.core import SimpleDirectoryReader
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter

from util.load_model import get_llm

get_llm()

documents = SimpleDirectoryReader(input_files=["./data/deepseek介绍.txt"]).load_data()

# 分割文本设置
splitter = TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)

# 进行标题的提取
title_extractor = TitleExtractor(nodes=5, node_template="请为以下文档生成一个简洁的标题: {context_str}", num_workers=5)

# 为每一个节点生成问题-默认的提示词是英文，手动添加提示词
question_prompt_template = """
    以下是参考内容：
    {context_str}
    
    请根据上述上下文信息，生成 {num_questions} 个该内容能够具体回答的问题，这些问题的答案最好是该内容独有的，不容易在其他地方找到。
    
    你也可以参考上下文中可能提供的更高层次的总结信息，结合这些总结，尽可能生成更优质、更具有针对性的问题。请用中文输出！
"""
# 进行问题的提取
questions_answered_extractor = QuestionsAnsweredExtractor(questions=3, node_template=question_prompt_template, num_workers=5)

async def main():
    # 官方建议使用摄取管道进行元数据提取
    # 将原始数据转换为可用于查询的结构化格式
    ingestion_pipeline = IngestionPipeline(transformations=[splitter, title_extractor, questions_answered_extractor])

    # 开始执行将原始数据转换为可索引的文档格式
    nodes = ingestion_pipeline.run(documents=documents, in_place=True, show_progress=True)

    print(nodes)

if __name__ == '__main__':
    asyncio.run(main())
