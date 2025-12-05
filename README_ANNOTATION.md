# SAM3 批量标注工具

这是一个基于SAM3的批量图像自动标注工具，包含服务端和前端界面。

## 功能特点

- 🚀 服务启动时自动加载SAM3模型到后台
- 📁 批量处理指定文件夹中的所有图像
- 🎯 支持多个文本提示词（class_id + name）
- 💾 自动在image_dir同级目录创建labels文件夹保存标注结果
- 🌐 友好的中文Web界面（端口8847）

## 安装依赖

```bash
conda activate SAM3
pip install fastapi uvicorn
```

注意：`pydantic`, `Pillow`, `torch`, `sam3` 已在SAM3环境中安装。

## 使用方法

### 1. 启动服务器

```bash
conda activate SAM3
cd /media/qzq/4t/AAA_myproject/SAM3/sam3
python server.py
```

服务器将在端口8847启动，模型会在启动时自动加载。

### 2. 打开前端界面

在浏览器中访问：
```
http://localhost:8847
```

### 3. 使用界面

1. **输入图像文件夹路径**：例如 `/path/to/your/images`
2. **添加提示词**：
   - 类别ID：例如 `1`
   - 类别名称：例如 `cat`
   - 点击"添加提示词"
   - 可以添加多个提示词（如 `2: dog`, `3: person`）
3. **开始标注**：点击"开始标注"按钮
4. **查看结果**：标注结果会保存在 `<image_dir>/../labels/` 文件夹中

## 输出格式

标注结果保存为PNG格式的二值掩码图像，命名规则：
```
<原图文件名>_<class_id>_<class_name>.png
```

例如：
- 原图：`image001.jpg`
- 提示词：`1: cat`
- 输出：`labels/image001_1_cat.png`

## API接口

### POST /annotate

请求体：
```json
{
  "image_dir": "/path/to/images",
  "prompts": [
    {"class_id": "1", "name": "cat"},
    {"class_id": "2", "name": "dog"}
  ]
}
```

响应：
```json
{
  "status": "completed",
  "total_images": 10,
  "labels_dir": "/path/to/labels",
  "results": [...]
}
```

## 技术栈

- **后端**：FastAPI + SAM3
- **前端**：HTML + JavaScript（原生）
- **模型**：SAM3 (Segment Anything Model 3)

## 注意事项

- 首次启动时模型加载需要一定时间，请耐心等待
- 确保有足够的GPU内存来加载SAM3模型
- 支持的图像格式：`.jpg`, `.jpeg`, `.png`, `.bmp`
- 每张图像的每个提示词会生成一个掩码文件
