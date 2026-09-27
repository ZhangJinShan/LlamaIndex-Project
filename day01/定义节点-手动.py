from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo


# 定义双向关系绑定函数：核心功能是给两个节点建立「NEXT/PREVIOUS」的双向关联
def link_bidirectional(a: TextNode, b: TextNode, a_note: str, b_note: str):
    """
    给两个 TextNode 建立双向的 NEXT/PREVIOUS 关系，并添加描述元数据
    Args:
        a: 第一个文本节点（作为「前序节点」）
        b: 第二个文本节点（作为「后序节点」）
        a_note: 从a指向b的关系描述（a的NEXT关系元数据）
        b_note: 从b指向a的关系描述（b的PREVIOUS关系元数据）
    """

    # 第一步：给节点a添加「NEXT（下一个）」关系 → 指向节点b
    # 意思是：a的下一个节点是b，并用metadata记录这个关系的描述
    a.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
        node_id=b.node_id,  # 关联的目标节点ID（b的唯一标识）
        metadata={"desc": a_note}  # 关系的描述元数据，方便后续追溯
    )
    # 第二步：给节点b添加「PREVIOUS（上一个）」关系 → 指向节点a
    # 意思是：b的上一个节点是a，形成双向绑定，元数据记录描述
    b.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
        node_id=a.node_id,  # 关联的目标节点ID（a的唯一标识）
        metadata={"desc": b_note}  # 关系的描述元数据
    )


node1 = TextNode(text="deepseek", id="1")
node2 = TextNode(text="chatgpt", id="2")

link_bidirectional(node1, node2, a_note="这是节点2", b_note="这是节点1")

nodes = [node1, node2]
print(nodes)