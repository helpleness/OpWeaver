import { expect, it } from "vitest";
import { groupAnalysis, relatedNodes } from "./analysis-view";
import type { AnalysisGraph } from "./code-types";
const graph = {
  id: "g",
  nodes: [
    {
      id: "a",
      label: "encoder/a",
      operator: "Relu",
      kind: "onnx_node",
      module_path: "",
    },
    {
      id: "b",
      label: "encoder/b",
      operator: "Relu",
      kind: "onnx_node",
      module_path: "",
    },
    {
      id: "c",
      label: "head",
      operator: "Add",
      kind: "onnx_node",
      module_path: "",
    },
    {
      id: "d",
      label: "isolated",
      operator: "Identity",
      kind: "onnx_node",
      module_path: "",
    },
  ],
  edges: [
    { id: "ab", source: "a", target: "b" },
    { id: "bc", source: "b", target: "c" },
    { id: "ac", source: "a", target: "c" },
  ],
} as AnalysisGraph;
it("groups view nodes without modifying source or making duplicate/self edges", () => {
  const before = JSON.stringify(graph),
    result = groupAnalysis(graph, "operator");
  expect(result.nodes).toHaveLength(3);
  expect(result.edges).toHaveLength(1);
  expect(result.nodes[0].details.group_members).toEqual(["a", "b"]);
  expect(JSON.stringify(graph)).toBe(before);
});
it("traces upstream and downstream while excluding disconnected nodes", () => {
  expect(relatedNodes(graph, "b", "upstream").sort()).toEqual(["a", "b"]);
  expect(relatedNodes(graph, "b", "downstream").sort()).toEqual(["b", "c"]);
  expect(relatedNodes(graph, "b", "both").sort()).toEqual(["a", "b", "c"]);
  expect(relatedNodes(graph, "b", "none")).toEqual([]);
});
it("module grouping keeps all original members for expansion", () => {
  expect(
    groupAnalysis(graph, "module").nodes.flatMap(
      (n) => n.details.group_members,
    ),
  ).toHaveLength(4);
  expect(groupAnalysis(graph, "none")).toBe(graph);
});
