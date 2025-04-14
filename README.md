# 📦 目标检测标注系统（类 Roboflow 项目管理）

本系统是一个轻量级目标检测标注平台，支持项目管理、图像上传与框选标注，可导出标准 JSON 数据格式，适合个人或小型团队使用。

---

## 🚀 快速启动

### 1. 安装依赖
```bash
pip install flask
```

### 2. 启动项目
```bash
python app.py
```

默认运行在：`http://127.0.0.1:5000`

---

## 🧱 项目结构说明
```
dataset_tool/
├── app.py                  # Flask 后端入口
├── templates/              # 前端页面
│   ├── project_list.html   # 项目主页
│   ├── project_detail.html # 项目详情（图像上传）
│   └── annotate.html       # 标注界面（Konva.js）
├── static/                 # 静态资源目录（预留）
├── data/                   # 数据目录（自动生成）
│   ├── images/             # 各项目上传图像目录
│   └── annotations/        # 各项目标注 JSON 文件
└── README.md               # 项目说明文件
```

---

## ✨ 功能概览

### 📁 项目管理
- 创建/管理多个项目
- 每个项目包含图像和标注信息

### 🖼️ 图像管理
- 拖拽上传或点击上传
- 缩略图自动预览

### 🎯 标注工具（Konva.js）
- 鼠标框选目标（矩形框）
- 手动输入类别
- 框可拖动、自动保存
- 加载已有标注记录

---

## 📤 标注数据格式（保存为 JSON）
每张图像保存一个对应 JSON，格式如下：
```json
[
  {"x": 100, "y": 120, "width": 60, "height": 40, "label": "cat"},
  {"x": 300, "y": 200, "width": 80, "height": 50, "label": "dog"}
]
```

后续可扩展导出为 COCO / VOC / YOLO 格式。

---

## 📸 示例截图

### 项目管理页面
![项目管理](https://via.placeholder.com/600x180?text=项目管理列表)

### 项目图像浏览
![图像上传](https://via.placeholder.com/600x180?text=图像上传与浏览)

### 图像标注界面
![图像标注](https://via.placeholder.com/600x180?text=标注界面-Konva绘制)

---

## 📌 后续扩展建议
- 类别管理：自定义分类列表 / 下拉选择
- 标注导出：支持 COCO / VOC / YOLO
- 自动标注：接入预训练模型（YOLOv5/v8）进行预标
- 用户管理：标注协作 + 权限系统

---
