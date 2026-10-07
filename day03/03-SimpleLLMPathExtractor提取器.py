from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor

from util.load_model import get_embed, get_llm

# 加载 llm 和 embedding 模型
llm = get_llm()
embed = get_embed()

# 加载文档
documents = SimpleDirectoryReader(
    input_files=["./data/小说.txt"]
).load_data()

# 创建提取规则
kg_extractor = SimpleLLMPathExtractor(
    llm=llm,
    max_paths_per_chunk=10,  # 控制从每个文档块(chunk)中最多提取多少条路径
    num_workers=4  # 并行数量
)
# 创建属性图
index = PropertyGraphIndex.from_documents(
    documents=documents,
    transformations=[kg_extractor],
    show_progress=True  # 显示提取进度
)
# 查看结果
triplets = index.property_graph_store.get_triplets(entity_names=["萧炎"])

for subj, rel, obj in triplets:
    # 解包成 (LabelledNode, Relation, LabelledNode)
    print(subj, "->" , rel, "->", obj)
    print("-" * 20)

