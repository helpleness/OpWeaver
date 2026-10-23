<script setup lang="ts">
import { ref } from "vue";
import { useLab } from "../stores/lab";
const lab = useLab();
const manifest = ref(""),
  root = ref(""),
  input = ref(lab.source?.inputs[0]?.name ?? "");
const batch = ref(2),
  dropLast = ref(false),
  color = ref("RGB"),
  layout = ref("NCHW"),
  dtype = ref("float32"),
  scale = ref(1 / 255);
const shape = lab.source?.inputs[0]?.shape;
const resize = ref(
  JSON.stringify(
    shape?.length === 4 &&
      typeof shape[2] === "number" &&
      typeof shape[3] === "number"
      ? [shape[2], shape[3]]
      : null,
  ),
);
const crop = ref("null"),
  mean = ref("[0,0,0]"),
  std = ref("[1,1,1]"),
  interpolation = ref("bilinear"),
  exif = ref(true),
  classes = ref("{}"),
  provenance = ref("not_established");
function config() {
  return {
    schema_version: "1.0",
    manifest_path: manifest.value,
    root: root.value,
    input_name: input.value,
    batch_size: batch.value,
    drop_last: dropLast.value,
    class_mapping: JSON.parse(classes.value),
    provenance: provenance.value,
    preprocessing: {
      schema_version: "1.0",
      resize: JSON.parse(resize.value),
      center_crop: JSON.parse(crop.value),
      color: color.value,
      layout: layout.value,
      dtype: dtype.value,
      scale: scale.value,
      mean: JSON.parse(mean.value),
      std: JSON.parse(std.value),
      interpolation: interpolation.value,
      exif_transpose: exif.value,
    },
  };
}
defineExpose({ config });
</script>
<template>
  <div
    class="image-dataset-editor"
    aria-label="Image classification dataset configuration"
  >
    <p class="muted small">
      Local PNG/JPEG · UTF-8 CSV with <code>path,label</code> columns. Label
      values map explicitly to model score columns. All source files and
      preprocessing are snapshotted.
    </p>
    <p class="muted small">
      Up to 4,096 images / 512 batches; 16 MiB per image, 256 MiB source files
      and 1 GiB tensors per dataset. Preparation is limited to 45 seconds.
    </p>
    <label
      >Image CSV manifest<input
        aria-label="Image CSV manifest"
        v-model="manifest"
        placeholder="C:\data\validation.csv"
    /></label>
    <label
      >Image root (default: CSV folder)<input
        aria-label="Image root directory"
        v-model="root"
    /></label>
    <label
      >Model input name<select aria-label="Image input name" v-model="input">
        <option
          v-for="item in lab.source?.inputs"
          :key="item.name"
          :value="item.name"
        >
          {{ item.name }} · {{ item.dtype }}
        </option>
      </select></label
    >
    <label
      >Label → model score column (JSON)<textarea
        aria-label="Image class mapping"
        rows="3"
        v-model="classes"
        placeholder='{"cat":0,"dog":1}'
      />
    </label>
    <div class="image-config-grid">
      <label
        >Batch size<input
          aria-label="Image batch size"
          type="number"
          min="1"
          max="64"
          v-model.number="batch" /></label
      ><label
        >DType<select aria-label="Image input dtype" v-model="dtype">
          <option>float32</option>
          <option>float16</option>
          <option>float64</option>
          <option>uint8</option>
        </select></label
      >
    </div>
    <label class="inline"
      ><input
        type="checkbox"
        aria-label="Image drop last batch"
        v-model="dropLast"
      />Explicitly drop incomplete final batch</label
    >
    <label
      >Resize [height,width] or null<input
        aria-label="Image resize"
        v-model="resize"
    /></label>
    <label
      >Center crop [height,width] or null<input
        aria-label="Image center crop"
        v-model="crop"
    /></label>
    <div class="image-config-grid">
      <label
        >Color<select aria-label="Image color order" v-model="color">
          <option>RGB</option>
          <option>BGR</option>
        </select></label
      ><label
        >Layout<select aria-label="Image layout" v-model="layout">
          <option>NCHW</option>
          <option>NHWC</option>
        </select></label
      >
    </div>
    <label
      >Pixel scale<input
        aria-label="Image pixel scale"
        type="number"
        step="any"
        v-model.number="scale"
    /></label>
    <label
      >Mean per channel (after color order)<input
        aria-label="Image mean"
        v-model="mean" /></label
    ><label
      >Std per channel<input aria-label="Image std" v-model="std"
    /></label>
    <p class="muted small">
      Tensor = (pixel × scale − mean) / std. Configure this for your model;
      ImageNet normalization is never assumed. RGB conversion expands grayscale
      and discards alpha.
    </p>
    <label
      >Resize interpolation<select
        aria-label="Image interpolation"
        v-model="interpolation"
      >
        <option>bilinear</option>
        <option>nearest</option>
        <option>bicubic</option>
      </select></label
    >
    <label class="inline"
      ><input
        type="checkbox"
        aria-label="Image EXIF orientation"
        v-model="exif"
      />Apply EXIF orientation</label
    >
    <label
      >Dataset provenance<select
        aria-label="Image dataset provenance"
        v-model="provenance"
      >
        <option value="not_established">
          Representativeness not established
        </option>
        <option value="user_declared">
          User declares representative coverage
        </option>
        <option value="synthetic_test">Synthetic test fixture</option>
      </select></label
    >
  </div>
</template>
<style scoped>
.image-config-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.inline {
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 7px;
}
.inline input {
  width: 14px;
  flex-shrink: 0;
}
.image-dataset-editor {
  padding-top: 6px;
}
</style>
