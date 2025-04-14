. `
🐱 AILabeler 是一个轻量级目标检测标注系统，专为 YOLO 系列模型设计，支持图像上传、标注框管理、类别设置、自动标注（YOLOv8）、导出多格式训练数据等功能。
项目已经发布至https://github.com/as501226107/AILabeler，后续会持续更新，欢迎 star

---

🔧 安装运行
# 安装依赖
pip install flask ultralytics opencv-python pillow
# 运行服务
python app.py
# 默认访问：http://localhost:5001



---

📌 常用接口说明| 路径 | 说明 | 
| --- | --- | 
| / | 项目列表页 | 
| /project/<项目名> | 项目详情（上传/预览图像） | 
| /annotate/<项目>/<图像> | 标注页面（主界面） | 
| /autolabel/<项目>/<图像> | 自动标注接口（YOLOv8） | 
| /export/<项目> | 数据导出页面 | 


---

🧠 自动标注说明自动标注使用 YOLOv8 模型进行推理。你可以根据需求替换模型文件：
model = YOLO("yolov8n.pt")  # 替换为自己的 yolov8.pt

并在 `
🐱 YOLabeler 标注系统YOLabeler 是一个轻量级目标检测标注系统，专为 YOLO 系列模型设计，支持图像上传、标注框管理、类别设置、自动标注（YOLOv8）、导出多格式训练数据等功能。

---

🚀 功能特性📂 项目管理 
- 创建/查看项目
 - 每个项目拥有独立图像、类别、标注数据
![](https://fastly.jsdelivr.net/gh/bucketio/img12@main/2025/04/14/1744621464030-ed433f90-3005-4728-8ae8-f7ed16ea4b0b.png)

![](https://fastly.jsdelivr.net/gh/bucketio/img9@main/2025/04/14/1744621476068-451a708c-2ee7-417e-a1f2-2696e4e3fb32.png)


![](https://fastly.jsdelivr.net/gh/bucketio/img7@main/2025/04/14/1744621502565-2a346717-6487-4e4f-9819-1a34c07c2113.png)

🖼 图像上传与预览 
- 支持批量上传图片（JPG/PNG）
 
- 图像状态展示：✅ 已标注 / ❌ 未标注
🧾 标注功能 
- 使用鼠标框选目标区域
 
- 每个框支持单独设置类别
 
- 右侧列表可编辑/删除每个框
 
- 自动保存为 JSON 标注数据

![](https://fastly.jsdelivr.net/gh/bucketio/img11@main/2025/04/14/1744621644619-cf17645d-0b4a-4f76-83b7-2be9eeb46b9f.png)

![](https://fastly.jsdelivr.net/gh/bucketio/img5@main/2025/04/14/1744621617545-a82c9673-c2f5-4cfa-843c-9106ea1bb3d8.png)

🔁 图像浏览 
- 支持上一张 / 下一张图片快捷切换
 
- 切图时自动加载对应标注

🧠 自动标注 
- 集成 YOLOv8 模型（默认 yolov8n.pt）
 
- 可识别图像中的所有物体
 
- 支持基于项目类别过滤保留框
![](https://fastly.jsdelivr.net/gh/bucketio/img15@main/2025/04/14/1744621593076-6687bdd2-d8b0-4e4b-960c-391836a87980.png)

![](https://fastly.jsdelivr.net/gh/bucketio/img16@main/2025/04/14/1744621538082-aa92e97e-8ba2-4e24-b43c-8d23c3451472.png)

📤 数据导出 
- 支持三种格式：YOLO / COCO / VOC
 
- 可生成 ZIP 包一键下载
⚙️ 类别管理 
- 项目级类别配置（支持增删改）
 
- 自动标注时仅保留已配置类别

![](https://fastly.jsdelivr.net/gh/bucketio/img1@main/2025/04/14/1744621683770-15b81a3d-8487-4c34-831e-a1e9bf5d0beb.png)


---

🧰 目录结构

YOLabeler/
├── app.py               # 主 Flask 应用
├── templates/           # HTML 模板（Jinja2）
├── static/              # 可选：CSS/JS 静态资源
├── data/
│   ├── images/          # 图像数据按项目分类存储
│   ├── annotations/     # 标注数据存储
│   ├── config/          # 每个项目类别配置
│   └── projects.json    # 所有项目注册信息



---

🔧 安装运行
# 安装依赖
pip install flask ultralytics opencv-python pillow
# 运行服务
python app.py
# 默认访问：http://localhost:5001



---

📌 常用接口说明| 路径 | 说明 | 
| --- | --- | 
| / | 项目列表页 | 
| /project/<项目名> | 项目详情（上传/预览图像） | 
| /annotate/<项目>/<图像> | 标注页面（主界面） | 
| /autolabel/<项目>/<图像> | 自动标注接口（YOLOv8） | 
| /export/<项目> | 数据导出页面 | 


---

🧠 自动标注说明自动标注使用 YOLOv8 模型进行推理。你可以根据需求替换模型文件：
model = YOLO("yolov8n.pt")  # 替换为自己的 yolov8.pt

并在 data/config/<项目>_classes.json` 中指定需要保留的类别，非该列表内的将被过滤。

---

📦 支持导出格式YOLO 格式： 
- 每张图片一个 `每张图片一个 .txt`
 
- 内容：`内容：<class> <x_center> <y_center> <width> <height>`（归一化）
COCO 格式： 
- 一个 `一个 annotations.json`
 
- 包含 images、annotations、categories 三部分
VOC 格式： 
- 每张图片一个 `每张图片一个 .xml` 文件，标准 Pascal VOC 格式


---

🧠 关键代码解释1. /annotate/<project>/<filename>` 路由用于渲染标注页面，加载图像与对应标注框，支持上下图切换：
  ```python
@app.route('/annotate/<project>/<filename>')def annotate(project, filename):    ... # 加载图像目录与标注    return render_template('annotate.html', ...)
```
2. /autolabel/<project>/<filename>` 路由使用 YOLOv8 推理当前图像，返回建议框：
  ```python
@app.route('/autolabel/<project>/<filename>')def autolabel_image(project, filename):    ... # 推理图像，生成预测框并渲染页面
```
3. saveBoxes()` 前端函数采集标注框坐标、类别，发送到 /save/<project>` 接口：

```python
function saveBoxes() {
  const result = boxes.map(b => ({ x: b.rect.x(), y: b.rect.y(), width: b.rect.width(), height: b.rect.height(), label: b.label }));
  fetch(`/save/${project}`, { method: 'POST', body: JSON.stringify({ filename, boxes: result }) })
}
```
4. 图像加载 + Konva 渲染逻辑前端绘图模块加载图像并按标注框显示：
```python
const image = new Image();
image.onload = () => {
  stage = new Konva.Stage(...);
  preloadBoxes.forEach(b => addBox(...));
  setupDrawing();
};
```


---

✅ 使用建议 
- 尽量使用 Chrome 浏览器
 
- 上传图像后建议立即标注保存（自动保存为 JSON）
 
- 自动标注可加快效率，但需人工复查每张图
 
- 切图前请先点击保存，防止未保存的修改丢失


---

📮 联系/建议欢迎提交 Issues 或 PR 改进系统 ❤️
