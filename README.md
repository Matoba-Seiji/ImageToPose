# ImageToPose

基于 MediaPipe 的单图姿态工具，支持 3ds Max Biped 重定向和 Maya 普通关节生成。

## 功能

- 3ds Max：全身与独立手部两个 Tab，支持图片选择、拖入和粘贴。
- 全身姿态写入目标 Biped，并保留一套普通 Bone 预览；重复生成替换旧预览。
- 近伸直手臂和腿部自动稳定，减少关键点误差导致的异常轴向扭转。
- 服务返回可靠面部三维点时估计头部朝向；缺少数据时保留原相对旋转。
- 独立手部支持左右手目标及输入手性转换，写入五指，不生成独立手部 Bone。
- 单次写入使用 Undo 事务，失败回滚，并提供最近一次写入的恢复。
- Maya：全身加手指、独立手部图片生成普通 joint hierarchy。

## 目录

```text
3dsmax_pose_tool/   Max 界面、HTTP 客户端和 Biped 求解
maya_pose_tool/     MediaPipe 推理、Maya 界面和关节生成
pose_server/       独立 HTTP 推理服务
tests/             数学、协议、骨架、Qt 流程和模拟回滚测试
tools/             服务运行包构建及压缩脚本
```

仓库仅包含源码，不包含模型权重、预编译运行包、场景文件、参考图片或用户缓存。Max 客户端不需要安装 MediaPipe；推理由独立服务完成。

## 推理服务

使用独立 Python 3.11 环境安装依赖：

```powershell
python -m pip install -r pose_server/requirements.txt
```

从官方 MediaPipe 资源获取 `pose_landmarker_full.task` 和 `hand_landmarker.task`，放到用户目录的 `.mediapipe/models/`，或设置 `MEDIAPIPE_POSE_MODEL`、`MEDIAPIPE_HAND_MODEL` 指向相应文件。模型使用须遵守其上游许可条款。

- [Pose Landmarker 文档](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python)
- [Hand Landmarker 文档](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python)

在仓库根目录启动：

```powershell
python -m pose_server.server --host 127.0.0.1 --port 18765
```

接口为 `GET /health`、`POST /v1/pose`、`POST /v1/hand-pose`，上传使用 multipart 的 `image` 字段。服务本身没有身份认证，请保持本机监听，不要直接暴露到公网。

Max 默认使用本机服务；通过 `POSE_SERVER_URL`、`HAND_POSE_SERVER_URL` 可配置受信任的远程服务。使用远程服务时，输入图片会发送到该地址。

## 在 3ds Max 中打开

在 Max 的 Python 脚本窗口执行，将路径替换为本机仓库目录：

```python
import importlib
import sys

root = r"C:\Tools\ImageToPose"
if root not in sys.path:
    sys.path.insert(0, root)
tool = importlib.import_module("3dsmax_pose_tool.pose_tool")
tool.show_pose_window()
```

主要面向 3ds Max 2025，Qt 导入兼容 PySide2/PySide6。全身目标通过所选 Biped 节点识别；未选择时场景必须只有一个 Biped。要求 3 节脊柱、1 节颈、常规四肢与五指三节结构，不支持额外动画层及 Figure/Footstep 等特殊模式。

全身写入帧在点击时固定。独立手部固定使用 `Bip001 L Hand` 或 `Bip001 R Hand`，识别返回时在当前帧写入，等待期间请勿切帧。拇指参考缓存保存在仓库内 `outputs/hand_references`。

自动肘膝稳定会修正弯曲平面，Biped 方向可能与原始预览 Bone 略有差别，结果中的 `projection_degrees` 记录该差异。不提供手动肘平面参数、单指写入、旧左手入口或从场景骨架重新应用接口。

## 在 Maya 中打开

Maya 模式在 Maya Python 环境内执行推理，需要为对应 `mayapy` 安装兼容的 MediaPipe、OpenCV、NumPy，并配置模型路径。

```python
import importlib
import sys

root = r"C:\Tools\ImageToPose"
if root not in sys.path:
    sys.path.insert(0, root)
tool = importlib.import_module("maya_pose_tool.mediapipe_image_to_skeleton")
tool.show_mediapipe_pose_window()
```

## 构建独立服务

```powershell
python -m pip install pyinstaller
powershell -ExecutionPolicy Bypass -File tools/build_pose_server.ps1 -Python C:\Python311\python.exe -ModelDirectory C:\Models\MediaPipe
python tools/package_pose_runtime.py
```

将完整的 `dist/jade_pose_server` 目录部署为仓库根目录的 `PoseRuntime`，或通过 `POSE_SERVER_EXE` 指定可执行文件。没有本地服务时，Max 客户端会尝试启动该运行包。也可运行 `start_pose_server.bat` 启动开发构建。

## 测试

在仓库根目录执行：

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Qt 流程测试在安装 PySide2/PySide6 时运行，否则跳过。事务测试使用模拟 Max API，不等同于真实 Max 场景验收。
