// @vitest-environment jsdom
import { beforeEach, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import LossChart from "./LossChart.vue";
import NodeInspector from "./NodeInspector.vue";
import TaskQualityCard from "./TaskQualityCard.vue";
import { useEditor } from "../stores/editor";

vi.mock("../api", () => ({
  rpc: vi
    .fn()
    .mockResolvedValue({ valid: true, issues: [], order: [], tensors: {} }),
}));
beforeEach(() => setActivePinia(createPinia()));
it("shows measured classification counts and separates percentage points from relative percent", () => {
  const reference = {
    status: "evaluated",
    samples: 12,
    classes: 6,
    correct: 8,
    incorrect: 4,
    top1_accuracy: 2 / 3,
    top5_accuracy: 1,
    top5_status: "evaluated",
  };
  const candidate = {
    ...reference,
    correct: 12,
    incorrect: 0,
    top1_accuracy: 1,
  };
  const wrapper = mount(TaskQualityCard, {
    props: {
      paired: {
        baseline_metrics: reference,
        optimized_metrics: candidate,
        accuracy_drop_percentage_points: -100 / 3,
        accuracy_change_percent: 50,
      },
    },
  });
  expect(wrapper.text()).toContain("8 / 4");
  expect(wrapper.text()).toContain("12 / 0");
  expect(wrapper.text()).toContain("-33.333 percentage points");
  expect(wrapper.text()).toContain("relative change 50%");
});
it("does not show fabricated accuracy or Top5 for unsupported class counts", () => {
  const absent = mount(TaskQualityCard, {
    props: { single: { status: "not_evaluated" } },
  });
  expect(absent.find("table").exists()).toBe(false);
  const wrapper = mount(TaskQualityCard, {
    props: {
      single: {
        status: "evaluated",
        samples: 3,
        classes: 2,
        correct: 1,
        incorrect: 2,
        top1_accuracy: 1 / 3,
        top5_accuracy: null,
        top5_status: "not_applicable_fewer_than_5_classes",
      },
    },
  });
  expect(wrapper.text()).toContain("Top5 is not applicable");
  expect(wrapper.findAll("tbody tr")[1].text()).toContain("—");
});
it("does not fabricate chart data before training", () => {
  const wrapper = mount(LossChart, { props: { metrics: [] } });
  expect(wrapper.find("polyline").exists()).toBe(false);
  expect(wrapper.text()).toContain("first experiment");
});
it("renders the actual received step range", () => {
  const metrics = [10, 11].map((step) => ({
    step,
    epoch: 2,
    batch: 1,
    loss: 1 / step,
    accuracy: 0.8,
    learning_rate: 0.01,
    elapsed_seconds: 2,
  }));
  const wrapper = mount(LossChart, { props: { metrics } });
  expect(wrapper.find("polyline").attributes("points")).toContain("52,");
  expect(wrapper.text()).toContain("11 steps");
});
it("inspector parameter input edits Graph IR", async () => {
  const store = useEditor();
  store.registry = [
    {
      id: "core.linear",
      label: "Linear",
      version: "1",
      category: "Layers",
      inputs: ["in"],
      outputs: ["out"],
      defaults: { in_features: 8, out_features: 16 },
      parameters: {
        properties: {
          in_features: { type: "integer" },
          out_features: { type: "integer" },
        },
      },
      description: "Affine",
      capabilities: {
        dtypes: ["float32"],
        devices: ["cpu"],
        autograd: true,
        torch_export: false,
        onnx: false,
        custom_kernel: false,
        backends: ["pytorch.eager"],
      },
    },
  ];
  store.addNode("core.linear");
  const wrapper = mount(NodeInspector);
  const fields = wrapper.findAll('input[type="number"]');
  await fields[1].setValue("3");
  await fields[1].trigger("change");
  expect(store.selectedNode?.params?.out_features).toBe(3);
});
