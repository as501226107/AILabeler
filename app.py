# Flask 应用主入口
import shutil
import uuid
import zipfile

from flask import Flask, request, render_template, jsonify, redirect, url_for, send_from_directory, send_file
import os, json
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = 'data'
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'images')
ANNOTATION_FOLDER = os.path.join(BASE_DIR, 'annotations')
PROJECTS_FILE = os.path.join(BASE_DIR, 'projects.json')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATION_FOLDER, exist_ok=True)
os.makedirs(BASE_DIR, exist_ok=True)

if not os.path.exists(PROJECTS_FILE):
    with open(PROJECTS_FILE, 'w') as f:
        json.dump([], f)

# ---------------- 页面 ----------------
@app.route('/')
def index():
    with open(PROJECTS_FILE) as f:
        projects = json.load(f)
    return render_template('project_list.html', projects=projects)

@app.route('/project/<project_name>')
def project_detail(project_name):
    image_dir = os.path.join(UPLOAD_FOLDER, project_name)
    image_files = os.listdir(image_dir) if os.path.exists(image_dir) else []
    return render_template('project_detail.html', project=project_name, images=image_files)

# ---------------- 接口 ----------------
@app.route('/create_project', methods=['POST'])
def create_project():
    name = request.form['name']
    with open(PROJECTS_FILE) as f:
        projects = json.load(f)
    if name not in projects:
        projects.append(name)
        with open(PROJECTS_FILE, 'w') as f:
            json.dump(projects, f)
        os.makedirs(os.path.join(UPLOAD_FOLDER, name), exist_ok=True)
        os.makedirs(os.path.join(ANNOTATION_FOLDER, name), exist_ok=True)
    return redirect(url_for('index'))

@app.route('/upload/<project_name>', methods=['POST'])
def upload(project_name):
    image = request.files['image']
    filename = secure_filename(image.filename)
    save_dir = os.path.join(UPLOAD_FOLDER, project_name)
    os.makedirs(save_dir, exist_ok=True)
    image.save(os.path.join(save_dir, filename))
    return 'OK'

@app.route('/save/<project_name>', methods=['POST'])
def save(project_name):
    data = request.get_json()
    filename = data['filename'] + '.json'
    with open(os.path.join(ANNOTATION_FOLDER, project_name, filename), 'w') as f:
        json.dump(data['boxes'], f)
    return 'Saved'

@app.route('/load/<project_name>/<filename>')
def load(project_name, filename):
    filepath = os.path.join(ANNOTATION_FOLDER, project_name, filename + '.json')
    if os.path.exists(filepath):
        with open(filepath) as f:
            return jsonify(json.load(f))
    return jsonify([])
@app.route('/annotate/<project>/<filename>')
def annotate(project, filename):
    import os

    image_dir = os.path.join('data/images', project)
    anno_dir = os.path.join('data/annotations', project)
    image_list = sorted(os.listdir(image_dir))

    # 上/下一张逻辑
    index = image_list.index(filename)
    prev_filename = image_list[index - 1] if index > 0 else None
    next_filename = image_list[index + 1] if index < len(image_list) - 1 else None

    # 标注加载
    os.makedirs(anno_dir, exist_ok=True)
    anno_path = os.path.join(anno_dir, filename.replace('.jpg', '.json').replace('.png', '.json'))
    if os.path.exists(anno_path):
        with open(anno_path) as f:
            preload_data = json.load(f)
    else:
        preload_data = None

    return render_template(
        'annotate.html',
        project=project,
        filename=filename,
        prev_filename=prev_filename,
        next_filename=next_filename,
        preload=json.dumps(preload_data) if preload_data else 'null'
    )

@app.route('/images/<project>/<filename>')
def serve_image(project, filename):
    return send_from_directory(os.path.join(UPLOAD_FOLDER, project), filename)

@app.route('/project/<project>/classes', methods=['GET', 'POST'])
def project_classes(project):
    class_file = f'data/config/{project}_classes.json'
    os.makedirs('data/config', exist_ok=True)
    if request.method == 'POST':
        classes = request.form['class_list'].splitlines()
        classes = [c.strip() for c in classes if c.strip()]
        with open(class_file, 'w') as f:
            json.dump(classes, f)
        return redirect(url_for('project_classes', project=project))

    if os.path.exists(class_file):
        with open(class_file) as f:
            class_text = '\\n'.join(json.load(f))
    else:
        class_text = ''
    return render_template('project_classes.html', project=project, class_text=class_text)
@app.route('/config/<project>_classes.json')
def get_project_classes(project):
    path = f"data/config/{project}_classes.json"
    if os.path.exists(path):
        return send_from_directory("data/config", f"{project}_classes.json")
    else:
        return jsonify([])  # 没有类别时返回空列表
# app.py：扩展 /export/<project> 接口支持 COCO / VOC
import zipfile, shutil, uuid, datetime
from flask import send_file

@app.route('/export/<project>', methods=['GET', 'POST'])
def export_project(project):
    from flask import request
    if request.method == 'GET':
        return render_template('export.html', project=project)

    fmt = request.form['format']
    image_dir = os.path.join('data/images', project)
    anno_dir = os.path.join('data/annotations', project)
    class_file = os.path.join('data/config', f'{project}_classes.json')
    export_id = uuid.uuid4().hex[:8]
    export_path = f'data/exports/{project}_{export_id}_{fmt}'
    os.makedirs(export_path, exist_ok=True)

    with open(class_file) as f:
        classes = json.load(f)
    class_map = {name: idx for idx, name in enumerate(classes)}

    coco = {
        "info": {"year": datetime.datetime.now().year, "version": "1.0", "description": project},
        "images": [], "annotations": [], "categories": [
            {"id": i, "name": name, "supercategory": "object"} for i, name in enumerate(classes)
        ]
    }
    voc_template = """<annotation>\n  <folder>{project}</folder>\n  <filename>{filename}</filename>\n  <size><width>{w}</width><height>{h}</height><depth>3</depth></size>\n  {objects}\n</annotation>"""
    obj_template = "<object><name>{name}</name><bndbox><xmin>{xmin}</xmin><ymin>{ymin}</ymin><xmax>{xmax}</xmax><ymax>{ymax}</ymax></bndbox></object>"

    ann_id = 1
    for idx, name in enumerate(os.listdir(anno_dir)):
        img_id = idx + 1
        imgname = name.replace('.json', '')
        with open(os.path.join(anno_dir, name)) as f:
            boxes = json.load(f)

        if fmt == 'yolo':
            w, h = 1, 1
            label_lines = []
            for b in boxes:
                cls = class_map.get(b['label'], -1)
                if cls < 0: continue
                cx = (b['x'] + b['width']/2) / w
                cy = (b['y'] + b['height']/2) / h
                bw = b['width'] / w
                bh = b['height'] / h
                label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            with open(os.path.join(export_path, imgname + '.txt'), 'w') as f:
                f.write('\n'.join(label_lines))

        elif fmt == 'coco':
            coco["images"].append({"id": img_id, "file_name": imgname, "width": 1, "height": 1})
            for b in boxes:
                cls = class_map.get(b['label'], -1)
                if cls < 0: continue
                coco["annotations"].append({
                    "id": ann_id, "image_id": img_id, "category_id": cls,
                    "bbox": [b['x'], b['y'], b['width'], b['height']], "area": b['width'] * b['height'],
                    "iscrowd": 0
                })
                ann_id += 1

        elif fmt == 'voc':
            w, h = 1, 1
            objs = "\n  ".join([
                obj_template.format(
                    name=b['label'], xmin=int(b['x']), ymin=int(b['y']),
                    xmax=int(b['x'] + b['width']), ymax=int(b['y'] + b['height'])
                ) for b in boxes if b['label'] in class_map
            ])
            xml = voc_template.format(project=project, filename=imgname, w=w, h=h, objects=objs)
            with open(os.path.join(export_path, imgname + '.xml'), 'w') as f:
                f.write(xml)

    if fmt == 'coco':
        with open(os.path.join(export_path, 'annotations.json'), 'w') as f:
            json.dump(coco, f, indent=2)

    zipname = f'{export_path}.zip'
    with zipfile.ZipFile(zipname, 'w') as zipf:
        for fname in os.listdir(export_path):
            zipf.write(os.path.join(export_path, fname), arcname=fname)
    shutil.rmtree(export_path)
    return send_file(zipname, as_attachment=True)


# app.py 中添加图像预览页路由
@app.route('/project/<project>/images')
def project_images(project):
    image_dir = os.path.join('data/images', project)
    anno_dir = os.path.join('data/annotations', project)

    if not os.path.exists(image_dir):
        return f"Project {project} not found.", 404

    images = sorted(os.listdir(image_dir))
    marked = {f.replace('.json', '') for f in os.listdir(anno_dir)} if os.path.exists(anno_dir) else set()

    return render_template('project_images.html', project=project, images=images, marked_set=marked)

from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # 可修改为自定义模型路径

@app.route('/autolabel/<project>/<filename>', methods=['GET'])
def autolabel_image(project, filename):
    import cv2
    from PIL import Image
    img_path = os.path.join('data/images', project, filename)
    image = cv2.imread(img_path)
    h, w = image.shape[:2]

    results = model(img_path)[0]  # 获取预测结果（第一张）
    boxes = []
    with open(f"data/config/{project}_classes.json") as f:
        allowed_classes = set(json.load(f))

    for box in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls_id = box
        label = model.names[int(cls_id)]
        if label not in allowed_classes:
            continue
        boxes.append({
            "x": float(x1), "y": float(y1),
            "width": float(x2 - x1), "height": float(y2 - y1),
            "label": label, "confidence": float(conf)
        })

    return render_template("annotate.html", project=project, filename=filename, preload=json.dumps(boxes))
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=5001)
