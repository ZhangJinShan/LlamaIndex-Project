from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex

from util.load_model import get_llm, get_embed

# 加载 llm 和 embedding 模型
llm = get_llm()
embed = get_embed()

# 加载文档
documents = SimpleDirectoryReader(input_files=["./data/小说.txt"]).load_data()

# 创建属性图索引
property_graph_index = PropertyGraphIndex.from_documents(documents=documents, show_progress=True)

retriever = property_graph_index.as_retriever(
    include_text=True,  # 包括与匹配路径的源块
    similarity_top_k=2,  # 向量 kg 节点检索的前 k 个
)
nodes = retriever.retrieve("萧炎的斗之力是多少？")

print(nodes)

query_engine = property_graph_index.as_query_engine(
    include_text=False,  # 包括与匹配路径的源块
    similarity_top_k=3,  # 向量 kg 节点检索的前 k 个
)
response = query_engine.query("萧炎的斗之力是多少？")
print("-" * 20)
print(response)

