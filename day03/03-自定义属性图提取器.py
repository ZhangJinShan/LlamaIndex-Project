from typing import List

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

prompt = """
            从输入文本中提取实体以及实体之间的语义关系，严格按固定格式输出，**每行一条关系**，格式：`实体1|关系|实体2`
            入参文本：{text}
            输出仅保留提取出来的关系，不要额外解释、不要多余文字、不要 markdown。
        """

def parse_function(llm_output: str) -> List[List[str]]:
    """
    基础解析函数 - 解析简单的三元组格式
    输入: "实体1|关系|实体2" 格式的文本
    输出: [["实体1", "关系", "实体2"], ...] 格式的列表
    """
    paths = []
    lines = llm_output.strip().split('\n')
    for line in lines:
        line = line.strip()
        print("line -> ", line)
        if not line or line.startswith('#'):
            continue
        # 分割实体和关系
        parts = line.split("|")
        if len(parts) == 3:
            entity1, relation, entity2 = [part.strip() for part in parts]
            if entity1 and relation and entity2:
                paths.append([entity1, relation, entity2])

    print("paths -> ", paths)

    return paths


kg_extractor = SimpleLLMPathExtractor(llm=llm, extract_prompt=prompt, parse_fn=parse_function)

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

