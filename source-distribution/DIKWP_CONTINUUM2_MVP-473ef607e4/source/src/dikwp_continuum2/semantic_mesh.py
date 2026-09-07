from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Set, Tuple

from .models import DIKWP_TYPES, SemanticNode, SemanticTransform, clamp


class SemanticMesh:
    """A small auditable DIKWP×DIKWP semantic graph.

    Nodes can have mixed D/I/K/W/P membership. Edges record observer-, context-,
    and direction-specific transformations. The graph deliberately does not
    privilege a D→I→K→W→P path.
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, SemanticNode] = {}
        self.transforms: List[SemanticTransform] = []
        self._adj: Dict[str, Set[str]] = defaultdict(set)

    def add_node(self, node: SemanticNode) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate node: {node.node_id}")
        if not node.dikwp_mix:
            raise ValueError("dikwp_mix must not be empty")
        invalid = set(node.dikwp_mix) - set(DIKWP_TYPES)
        if invalid:
            raise ValueError(f"invalid DIKWP types: {sorted(invalid)}")
        total = sum(node.dikwp_mix.values())
        if total <= 0:
            raise ValueError("dikwp_mix weights must sum to a positive value")
        normalized = {key: clamp(value / total) for key, value in node.dikwp_mix.items()}
        self.nodes[node.node_id] = SemanticNode(
            node_id=node.node_id,
            text=node.text,
            dikwp_mix=normalized,
            observer=node.observer,
            context=node.context,
            provenance=node.provenance,
        )

    def add_transform(self, transform: SemanticTransform) -> None:
        if transform.source_node not in self.nodes or transform.target_node not in self.nodes:
            raise ValueError("transform endpoints must exist")
        if transform.source_type not in DIKWP_TYPES or transform.target_type not in DIKWP_TYPES:
            raise ValueError("invalid DIKWP transform type")
        self.transforms.append(transform)
        self._adj[transform.source_node].add(transform.target_node)

    def transform_coverage(self) -> float:
        pairs = {(edge.source_type, edge.target_type) for edge in self.transforms}
        return len(pairs) / 25.0

    def type_pair_counts(self) -> Dict[str, int]:
        counts = Counter(f"{edge.source_type}->{edge.target_type}" for edge in self.transforms)
        return dict(sorted(counts.items()))

    def observers(self) -> Set[str]:
        return {node.observer for node in self.nodes.values()}

    def strongly_connected_components(self) -> List[Set[str]]:
        # Tarjan's algorithm, implemented locally to keep the package stdlib-only.
        index = 0
        indices: Dict[str, int] = {}
        lowlink: Dict[str, int] = {}
        stack: List[str] = []
        on_stack: Set[str] = set()
        components: List[Set[str]] = []

        def visit(vertex: str) -> None:
            nonlocal index
            indices[vertex] = index
            lowlink[vertex] = index
            index += 1
            stack.append(vertex)
            on_stack.add(vertex)

            for nxt in self._adj.get(vertex, set()):
                if nxt not in indices:
                    visit(nxt)
                    lowlink[vertex] = min(lowlink[vertex], lowlink[nxt])
                elif nxt in on_stack:
                    lowlink[vertex] = min(lowlink[vertex], indices[nxt])

            if lowlink[vertex] == indices[vertex]:
                component: Set[str] = set()
                while True:
                    member = stack.pop()
                    on_stack.remove(member)
                    component.add(member)
                    if member == vertex:
                        break
                components.append(component)

        for vertex in self.nodes:
            if vertex not in indices:
                visit(vertex)
        return components

    def network_compatibility(self) -> Dict[str, object]:
        components = self.strongly_connected_components()
        largest = max((len(c) for c in components), default=0)
        coverage = self.transform_coverage()
        observer_count = len(self.observers())
        has_cycle = any(len(c) > 1 for c in components) or any(
            edge.source_node == edge.target_node for edge in self.transforms
        )
        leakage = 0.0
        if coverage < 0.6:
            leakage += 0.4
        if not has_cycle:
            leakage += 0.35
        if observer_count < 3:
            leakage += 0.25
        return {
            "transform_coverage": round(coverage, 4),
            "observer_count": observer_count,
            "largest_strong_component": largest,
            "has_cycle": has_cycle,
            "hierarchy_leakage": round(min(1.0, leakage), 4),
            "verdict": "NETWORK_COMPATIBLE" if leakage <= 0.25 else "REVIEW_NETWORK_STRUCTURE",
        }


def build_continuity_demo_mesh() -> SemanticMesh:
    mesh = SemanticMesh()
    nodes = [
        SemanticNode("n1", "延续不是复制，而是因果历史是否仍能影响未来行动", {"K": 0.5, "W": 0.25, "P": 0.25}, "段玉聪本人", "身份连续", "purpose interview"),
        SemanticNode("n2", "神经过程是否保持无破坏重叠", {"D": 0.4, "I": 0.3, "K": 0.3}, "神经科学团队", "渐进替换", "neural measurements"),
        SemanticNode("n3", "任何不可逆扫描不得被宣传为已证明本人继续存在", {"W": 0.5, "P": 0.4, "K": 0.1}, "伦理委员会", "破坏性上传", "governance rule"),
        SemanticNode("n4", "家庭是否仍承认其承诺、关系与责任", {"I": 0.25, "K": 0.2, "W": 0.35, "P": 0.2}, "家属与长期合作者", "关系连续", "relational testimony"),
        SemanticNode("n5", "数字代理能否在未知问题上重现价值冲突中的选择", {"D": 0.25, "I": 0.25, "K": 0.3, "P": 0.2}, "独立评测组", "心理连续", "blind challenge"),
        SemanticNode("n6", "未来后继体拥有拒绝被复制、重写或合并的程序权", {"W": 0.45, "P": 0.45, "I": 0.1}, "潜在数字后继体", "主体权利", "successor representation"),
        SemanticNode("n7", "身体内感受和情绪调节是否仍参与目的形成", {"D": 0.3, "I": 0.2, "K": 0.25, "W": 0.25}, "临床与体验研究组", "具身连续", "interoception tests"),
        SemanticNode("n8", "法律主体是否随密钥、记忆与责任谱系迁移", {"K": 0.3, "W": 0.3, "P": 0.4}, "法律受托人", "法律连续", "continuity trust"),
        SemanticNode("n9", "允许多个身份理论并存，不用单一总分封闭争议", {"K": 0.35, "W": 0.35, "P": 0.3}, "DIKWP-MESH²审计器", "概念解缚", "system rule"),
        SemanticNode("n10", "系统必须保留现象意识残差而不是宣布上传成功", {"K": 0.3, "W": 0.45, "P": 0.25}, "意识科学顾问", "意识连续", "indicator framework"),
    ]
    for node in nodes:
        mesh.add_node(node)

    # Ensure all 25 ordered type pairs are represented in an observer- and context-specific way.
    ordered_pairs = [(a, b) for a in DIKWP_TYPES for b in DIKWP_TYPES]
    for index, (source_type, target_type) in enumerate(ordered_pairs):
        source = nodes[index % len(nodes)]
        target = nodes[(index * 3 + 1) % len(nodes)]
        mesh.add_transform(
            SemanticTransform(
                transform_id=f"t{index+1:02d}",
                source_node=source.node_id,
                target_node=target.node_id,
                source_type=source_type,
                target_type=target_type,
                observer=target.observer,
                context=f"continuity-transform-{source_type}{target_type}",
                rationale="将一种语义资源转化为另一种资源，用于检验连续性、约束行动或提出新观测。",
                confidence=0.72 + (index % 5) * 0.04,
            )
        )
    return mesh
