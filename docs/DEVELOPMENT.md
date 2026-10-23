# 开发与部署

## Windows 11

依赖：Go 1.25+、Node.js 20.19+、Python 3.10+（推荐 3.13）、Microsoft Edge WebView2 Runtime。Wails v2.17.0 用 Go WebView2 loader；Windows 本项目构建不需要把 GPU 编译工具链装入 Go 主进程。

当前机器本地 `.tools/go` 是校验过官方 SHA256 的 Go 1.27.3，`.tools/bin/wails.exe` 是固定 Wails 2.17.0。没有更改系统 PATH。早期 Wails 2.10.2 在 Go 1.27 上的绑定生成出现 `package ... without types`；更新 Wails 及 x/tools 后实际构建通过，不能混用旧 CLI。

```powershell
.\scripts\bootstrap.ps1 -DownloadGo
.\scripts\dev.ps1
```

脚本创建 `.venv`，默认从官方 CPU wheel 索引安装 PyTorch；已有 PyTorch 时不替换。`-UseSystemTorch` 只在新建 venv 时启用 system-site-packages。脚本执行失败会停止，不将后续步骤成功掩盖前面的失败。

如 PowerShell 执行策略禁止本地脚本，可阅读脚本后在单次进程使用 `powershell -ExecutionPolicy Bypass -File .\scripts\dev.ps1`，或手动运行下列等价命令；无需永久修改系统策略：

```powershell
$env:PATH = "$PWD\.tools\go\bin;$PWD\.tools\bin;$env:PATH"
$env:PYTHONPATH = "$PWD\worker"
$env:OPWEAVER_PYTHON = "$PWD\.venv\Scripts\python.exe"
$env:OPWEAVER_HOME = "$PWD"
wails dev
```

如使用独立 Python 环境，将 OPWEAVER_PYTHON 设为 python.exe 的绝对路径。Worker 寻找顺序：该变量 → 资源目录及其祖先目录中的 .venv → PATH 上 python。因此直接从 build/bin 启动也可复用仓库 .venv。OPWEAVER_HOME 指向包含 worker/examples 的资源根目录；未设置时从当前目录/可执行文件父目录向上查找。

## 构建与分发

```powershell
.\scripts\build.ps1
```

输出 `build/bin/OpWeaverStudio.exe`，前端静态资源嵌入 exe；同目录复制 worker 源码、examples、pyproject、许可证。目标电脑仍需 Python/PyTorch/Pydantic/ONNX/ONNX Runtime CPU/NumPy（版本约束见 pyproject.toml）。可在分发目录执行 `python -m venv .venv`，安装 CPU torch 后 `pip install .`。也可指向独立 OPWEAVER_PYTHON。P0-C 本机实测 ONNX Runtime 1.31.0，不要求 CUDA；依赖不可用时 Lab 如实显示 unavailable。

没有自动打包 Python、没有代码签名、没有 Windows installer。`build/windows/` 保存 Wails 的应用元信息、manifest 和图标源。运行中的 exe 在 Windows 下不能覆盖，重建前关闭测试应用。

## 浏览器调试与 E2E

```powershell
# 使用和桌面完全相同的 Go/Python 后端，内嵌前端
.\scripts\dev.ps1 -Mode server
# 浏览器打开 http://127.0.0.1:34115

# 若需要前端热更新，另一个终端：
cd frontend
npm run dev
# http://127.0.0.1:5173，Vite 将 /api 代理到 Go
```

浏览器适配器没有原生目录对话框，可直接输入本地项目路径。不要与 `wails dev` 同时占用 34115。该端口只监听 loopback，不是远程 Worker 服务。停止开发服务用 Ctrl+C，Go 会尝试保存/取消正在运行的训练。

```powershell
# 本地 Go 服务已启动时；默认使用已安装 Chrome
cd frontend
npm run test:e2e
```

无 Chrome 时修改 `playwright.config.ts` 的 channel 或使用 `npx playwright install chromium` 并移除 channel。截图/trace 写入根目录 test-results，默认 gitignored。原生 Wails UI 的人工验收单独记录，浏览器 E2E 不能代替所有 WebView2 特定行为。

## 测试和类型生成

```powershell
.\scripts\test.ps1
```

其中 `OPWEAVER_INTEGRATION=1` 开启真实 Python 子进程测试（包括 kill/restart）。单独 `go test ./...` 时此测试会明确 skip，避免未安装 Python 的机器得到误导的失败。

```powershell
$env:PYTHONPATH = 'worker'
.\.venv\Scripts\python.exe scripts\generate_schema.py
cd frontend
npm run schema
npm run build
```

不要手工编辑 generated 下的五组类型。修改 Design IR、AnalysisGraph、SourceMapping、GraphImportResult 或 TrainingConfig 后一并生成 Schema/类型。Python 测试验证 Schema 没有漂移。

## P0-A 开发入口

Monaco 0.57 使用 `editor/editor.api.js` 和独立 Python 注册，编辑器及 diff 延迟导入，语言 worker 本地打包，不访问 CDN。生产构建的 Monaco chunk 较大（约 2.76 MB 未压缩），Vite 会提示体积警告；不是类型/构建失败。Windows WebView2 使用 classic textarea（editContext=false），避免输入/光标 API 差异。没有自动启动 Python LSP，诊断来自 AST 和实际捕获。

Python 模块：source.py（AST/映射/补丁）、codegen.py、capture.py、code_runner.py、code_tasks.py。前端：stores/code.ts、ModelWorkbench、PythonEditor、SourceDiff、AnalysisInspector。Go workspace.go 负责原子源码写入和 Hash 冲突，不复制 AST 逻辑。

可单独运行 `python -m pytest worker/tests/test_code.py -q`。完整 `npm run test:e2e` 覆盖 P0-A 和原训练流程；热更新验收可设置 `$env:OPWEAVER_E2E_URL='http://127.0.0.1:5173'`，最终验收应使用重新构建的 34115 内嵌前端。

纯 Python 模型示例在 examples/pytorch，不依赖联网下载权重。默认捕获仅 CPU/eval，输入 JSON 每项包括 shape/dtype，可选 design_shape；构造参数是 JSON 对象。打开文件不执行，Parse 启动显式独立子进程。捕获 stdout/stderr 和失败报告在项目 .opweaver/captures 中。

## Linux / WSL2 与 GPU

Python 执行引擎可独立运行，不依赖桌面：

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
PYTHONPATH=worker python scripts/demo.py --residual
pytest -q
```

CUDA 机器应按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 安装与驱动匹配的 wheel。验证 `torch.cuda.is_available()` 后再选择 CUDA；CUDA 不可用时明确报错，CPU 仍可用。

WSL2 的 Python 与 Windows Python 是不同环境，Windows exe 不能直接把 Linux python 路径当作 OPWEAVER_PYTHON。首版可在 WSL2 内运行 headless Worker/测试，原生 Windows GUI 默认启动 Windows Python。跨 WSL/远程执行将走后续 transport adapter。Triton/TensorRT/DALI 等建议后续以 Linux/WSL2 后端落地，当前没有要求安装它们。

Linux 原生 Wails 桌面还需要 GTK/WebKit2GTK/GCC 等系统依赖，参阅 [Wails 安装说明](https://v2.wails.io/docs/gettingstarted/installation/)；本轮未进行 Linux GUI 构建验收。

## 常见问题

- **ModuleNotFoundError**：用 OPWEAVER_PYTHON 指定的解释器安装 `pip install -e .`，不要误装到另一 Python。
- **Worker disconnected**：检查 Console stderr，点击顶部 Reconnect；恢复最后 durable checkpoint。不要把进程隔离视为 sandbox。
- **Graph mismatch on resume**：Checkpoint 与当前 IR 必须一致，包括图 metadata。使用 checkpoint 旁的 graph 快照恢复同一模型；UI 的“一键恢复历史图”留到 P2。
- **Already finished epochs**：Resume 的 epoch 是总目标，须大于 Checkpoint 已完成的 epoch。
- **Shape pending**：先连接必需端口，查看 Problems 定位缺失/不匹配项。
- **大量旧 graph/layout 文件**：首版保留不可变修订，没有自动 GC；不要删除 manifest 当前引用的文件。
- **本地项目复制到其他目录**：manifest 路径相对，但 SQLite 目前记录绝对 checkpoint 路径；迁移目录后的 Artifact 重定位需要 P2 支持。
- **训练中关闭窗口**：Go 发送 shutdown 并等待有限时间；GPU kernel 卡死可能被强制结束，此时只能恢复上一 durable checkpoint。

## 版本与可重复性

前端 `package-lock.json` 和 `go.sum` 已锁定解析结果；Python 用 pyproject 兼容范围，`requirements-tested.txt` 记录本轮 CPU 可安装的版本组合。CUDA wheel 根据宿主驱动单独选择。精确恢复测试在同一 CPU/PyTorch 环境比对权重；不同版本或不同设备不承诺 bitwise 相同。

## P1-A 独立 Olive 环境

显式执行 `scripts/setup_olive.ps1`；它创建独立 `.tools/olive-env`，不会装进 Worker，也不会被应用启动流程调用。环境的 `Scripts/python.exe` 配到 Lab 的 Microsoft Olive 面板；安装锁定见 requirements-olive.txt，完整依赖记录在环境 installed-requirements.txt。当前 0.13.x API，实际验证 0.13.0，本地 ONNX/CPU 两个 Pass。扩展点和限制见 [P1-A](P1A_IMPLEMENTATION.md)。

`OPWEAVER_OLIVE_PYTHON` 可覆盖自动化测试的独立解释器路径（Python/Playwright 也接受 venv 目录）；Go 集成测试请传 python.exe 文件。未安装独立环境时真实 Olive 自动化用例会跳过，ORT 和其他回归照常运行；发布验收必须安装并确认没有跳过。`scripts/verify_olive.py <新验收目录> <Olive环境>` 是离线 CLI 冒烟脚本，不代替桌面验收。`scripts/test.ps1` 仍运行全部 Python/前端/Go，`npm run test:e2e` 通过本地 --serve 启动的应用 HTTP 适配器执行真实工作流。

## P0-B 导入模块（历史入口）

依赖 ONNX >=1.22,<2，本机实测 1.22.0；bootstrap 的 pip install -e '.[test]' 安装。CPU 即可解析，未引入 ONNX Runtime。新机器须用同一个 OPWEAVER_PYTHON 安装依赖，不能仅替换 exe。

importers.py 注册格式，model_artifact.py 定义报告，onnx_import.py 提取真实结构，model_tasks.py 复用调度，model_files.py 流式复制，import_runner.py 执行。ModelBrowser/ModelInspector/useModels 复用 GraphCanvas，analysis-view 只做视图分组/关系遍历。

```powershell
$env:PYTHONPATH = 'worker'
.\.venv\Scripts\python.exe scripts/generate_onnx_examples.py
.\.venv\Scripts\python.exe -m pytest worker/tests/test_import.py -q
```

报告在 .opweaver/imports/<id>/result.json，日志同目录，复制件和外部权重在 model/；PyTorch 副本在项目 src，可回到已有编辑器。类型生成现有五组，不手改 generated。Go App→Worker→ONNX 子进程也有真实测试，test.ps1 用 -count=1 避免 Go 缓存忽略 Python 文件修改。

## P1-B 开发与验收

执行 scripts/generate_quantization_examples.py 生成合成 CNN / 32 批校准 / 8 批独立评估 NPZ；fixture.json 使用相对路径，映射 image → pixels。worker/tests/test_quantization.py 实跑 MinMax/Percentile/Entropy、QDQ/QOperator、显式预处理导出和 lineage、共享数据校验/重叠/缺失、Top1 和质量 gate；不向 Worker 安装 Olive。scripts/test.ps1 含 Go 真实静态量化 API 集成；frontend E2E 含完整 NPZ→量化→导出→重载→Inspector。详细限制见 P1B_IMPLEMENTATION.md。
