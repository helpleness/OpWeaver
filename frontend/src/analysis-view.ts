import type { AnalysisGraph, AnalysisNode } from "./code-types";

// View projection only. The authoritative imported graph is never rewritten.
export function groupAnalysis(
  graph: AnalysisGraph,
  mode: "none" | "operator" | "module",
): AnalysisGraph {
  if (mode === "none") return graph;
  const groups = new Map<string, AnalysisNode[]>();
  for (const node of graph.nodes) {
    const name =
      mode === "operator"
        ? node.operator
        : node.label.includes("/")
          ? node.label.split("/").slice(0, -1).join("/")
          : node.module_path.includes(".")
            ? node.module_path.split(".").slice(0, -1).join(".")
            : node.kind;
    const key = name || node.operator;
    const members = groups.get(key) ?? [];
    members.push(node);
    groups.set(key, members);
  }
  const owner = new Map<string, string>();
  const nodes: AnalysisNode[] = [];
  for (const [label, members] of groups) {
    const id = `group_${nodes.length}`;
    members.forEach((n) => owner.set(n.id, id));
    nodes.push({
      id,
      label: `${label} · ${members.length}`,
      operator: "analysis.group",
      kind: "group",
      module_path: "Collapsed group",
      parameters: { members: members.map((n) => n.id) },
      inputs: ["in"],
      outputs: ["out"],
      tensors: {},
      details: { group_members: members.map((n) => n.id) },
    });
  }
  const seen = new Set<string>();
  const edges = graph.edges.flatMap((e) => {
    const source = owner.get(e.source)!,
      target = owner.get(e.target)!;
    const key = `${source}:${target}`;
    if (source === target || seen.has(key)) return [];
    seen.add(key);
    return [
      {
        id: `group_edge_${seen.size}`,
        source,
        target,
        source_port: "out",
        target_port: "in",
      },
    ];
  });
  return { ...graph, id: `${graph.id}:${mode}`, nodes, edges };
}

export function relatedNodes(
  graph: AnalysisGraph,
  selected: string | null,
  direction: "none" | "upstream" | "downstream" | "both",
) {
  if (!selected || direction === "none") return [];
  const result = new Set([selected]);
  for (const reverse of direction === "both"
    ? [true, false]
    : [direction === "upstream"]) {
    const adjacency = new Map<string, string[]>();
    for (const edge of graph.edges) {
      const from = reverse ? edge.target : edge.source,
        to = reverse ? edge.source : edge.target;
      const list = adjacency.get(from) ?? [];
      list.push(to);
      adjacency.set(from, list);
    }
    const queue = [selected],
      visited = new Set(queue);
    for (let i = 0; i < queue.length; i++)
      for (const next of adjacency.get(queue[i]) ?? [])
        if (!visited.has(next)) {
          visited.add(next);
          result.add(next);
          queue.push(next);
        }
  }
  return [...result];
}
