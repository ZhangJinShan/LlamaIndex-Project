import asyncio
from typing import List

from llama_index.core import get_response_synthesizer, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.workflow import (
    Event,
    Workflow,
    Context,
    step,
    StartEvent,
    StopEvent
)
from llama_index.retrievers.bm25 import BM25Retriever
from nltk.data import retrieve

from util.load_model import get_llm, get_embed


# 定义工作流中的事件类型
class QueryEvent(Event):
    """查询事件"""
    query: str

class VectorRetrievalEvent(Event):
    """向量检索事件"""
    nodes: List[NodeWithScore]
    query: str

class BM25RetrievalEvent(Event):
    """BM25关键词检索事件"""
    nodes: List[NodeWithScore]
    query: str

class PostProcessorEvent(Event):
    """后处理事件"""
    nodes: List[NodeWithScore]
    query: str


class ResponseEvent(Event):
    """响应事件"""
    response: str
    source_nodes: List[NodeWithScore]

class RAGWorkflow(Workflow):
    """RAG工作流"""
    def __init__(self, retriever: VectorIndexRetriever,
                 bm25_retriever: BM25Retriever):
        super().__init__()
        self.retriever = retriever
        self.bm25_retriever = bm25_retriever
        # 节点后处理器
        self.postprocessor = SimilarityPostprocessor(similarity_cutoff=0.5)
        # 响应合成器
        self.response_synthesizer = get_response_synthesizer()

    @step
    async def query_step(self, ctx: Context, ev: StartEvent) -> QueryEvent:
        """
        步骤一：处理用户查询
        """
        query = ev.query
        print(f"🔍 接收查询: {query}")
        processed_query = query.strip()
        return QueryEvent(query=processed_query)

    @step
    async def vector_retrieval_step(self, ctx: Context, ev: QueryEvent) -> VectorRetrievalEvent | None:
        """
        步骤二：向量检索
        """
        print(f"📚 开始检索相关文档...")
        query = ev.query
        # 创建查询束
        query_bundle = QueryBundle(query_str=query)
        # 执行检索
        retrieved_nodes = await self.retriever.aretrieve(query_bundle)
        return VectorRetrievalEvent(nodes=retrieved_nodes, query=query)

    @step
    async def bm25_retrieval_step(self, ctx: Context, ev: QueryEvent) -> BM25RetrievalEvent:
        """
        步骤二：BM25关键词检索
        """
        print(f"📚 bm25开始检索相关文档...")
        query = ev.query
        query_bundle = QueryBundle(query_str=query)
        nodes = await self.bm25_retriever.aretrieve(query_bundle)
        return BM25RetrievalEvent(nodes=nodes, query=query)

    @step
    async def postprocessor_step(self, ctx: Context,
                                 ev: VectorRetrievalEvent,
                                 bm25_ev: BM25RetrievalEvent) -> PostProcessorEvent:
        """
        步骤三：对检索结果进行后处理
        """
        print(f"🔧 开始后处理检索结果...")
        # 等待 Vector 检索事件结果
        vector_events = ctx.collect_events(ev, [VectorRetrievalEvent])
        print(f"✅ 已收集Vector检索事件")

        # 等待 BM25 检索事件结果
        bm25_events = ctx.collect_events(bm25_ev, [BM25RetrievalEvent])
        print(f"✅ 已收集BM25检索事件")

        # 合并所有检索结果
        all_nodes = []
        query = ""
        if vector_events:
            all_nodes.extend(vector_events[0].nodes)
            query = vector_events[0].query
            print(f"  - Vector检索: {len(vector_events[0].nodes)} 个节点")

        if bm25_events:
            all_nodes.extend(bm25_events[0].nodes)
            query = bm25_events[0].query
            print(f"  - BM25检索: {len(bm25_events[0].nodes)} 个节点")

        if not all_nodes:
            print("⚠️  没有找到任何检索结果")
            return PostProcessorEvent(nodes=[], query=query)

        print(f"🔄 开始后处理 {len(all_nodes)} 个文档片段...")

        # 创建查询束用于后处理
        query_bundle = QueryBundle(query_str=query)
        # 执行后处理（去重、过滤、重排序等）
        processed_nodes = self.postprocessor.postprocess_nodes(nodes=all_nodes, query_bundle=query_bundle)
        print(f"✅ 后处理完成，保留 {len(processed_nodes)} 个高质量文档片段")

        # 打印每个节点的相似度分数
        for i, node in enumerate(processed_nodes[:3]):  # 只显示前3个
            score = node.score if node.score else 0
            print(f"  - 文档片段 {i + 1}: 相似度 {score:.3f}")

        return PostProcessorEvent(nodes=processed_nodes, query=ev.query)

    @step
    async def response_synthesis_step(self, ctx: Context, ev: PostProcessorEvent) -> ResponseEvent:
        """
        步骤四：基于检索到的上下文生成最终答案
        """
        print(f"🤖 开始生成答案...")
        if not ev.nodes:
            return ResponseEvent(response="抱歉，没有找到相关信息来回答您的问题。", source_nodes=[])

        # 创建查询束
        query_bundle = QueryBundle(query_str=ev.query)
        # 使用响应合成器生成答案
        response = await self.response_synthesizer.asynthesize(query=query_bundle, nodes=ev.nodes)
        print(f"✅ 答案生成完成")
        return ResponseEvent(response=str(response), source_nodes=ev.nodes)

    @step
    async def response_step(self, ctx: Context, ev: ResponseEvent) -> StopEvent:
        """
        步骤五：输出响应
        """
        print(f"📃 输出响应...")
        response = ev.response
        source_nodes = ev.source_nodes
        print(f"📃 响应: {response}")
        print(f"📃 源节点: {source_nodes}")
        return StopEvent(result={
            "response": str(response),
            "source_nodes": source_nodes,
            "metadata": {
                "num_sources": len(ev.source_nodes),
                "query": ev.response
            }
        })

async def main():
    # 主函数
    # 1. 准备数据和索引（这里使用示例数据）
    print("📖 正在构建向量索引...")

    # 加入大模型和嵌入向量
    get_llm()
    get_embed()

    # 加载文件
    documents = SimpleDirectoryReader(input_files=["./data/小说.txt"]).load_data()
    # 初始化节点解析器
    splitter = SentenceSplitter(chunk_size=512)
    nodes = splitter.get_nodes_from_documents(documents)
    # 创建向量索引对象
    index = VectorStoreIndex.from_documents(documents=documents)
    # 创建向量检索器
    retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
    # 创建BM25检索器
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=3)

    print("✅ 向量索引构建完成")

    # 2. 创建并运行工作流
    workflow = RAGWorkflow(retriever=retriever, bm25_retriever=bm25_retriever)
    # 测试查询
    test_queries = [
        "萧炎的爸爸是谁？",
        "萧炎的妹妹是谁？"
    ]

    for query in test_queries:
        print(f"\n{'=' * 50}")
        print(f"🎯 测试查询: {query}")
        print(f"{'=' * 50}")

        # 运行工作流
        result = await workflow.run(query=query)

        # 显示结果
        print(f"\n📝 生成的答案:")
        print(f"{result['response']}")
        print(f"\n📊 元数据:")
        print(f"- 使用了 {result['metadata']['num_sources']} 个文档片段")
        print(f"- 原始查询: {result['metadata']['query']}")




if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main())
