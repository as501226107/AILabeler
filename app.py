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
    return render_template('annotate.html', project=project, filename=filename)
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

    # 加载类别映射
    with open(class_file) as f:
        classes = json.load(f)
    class_map = {name: idx for idx, name in enumerate(classes)}

    for name in os.listdir(anno_dir):
        imgname = name.replace('.json', '')
        with open(os.path.join(anno_dir, name)) as f:
            boxes = json.load(f)

        if fmt == 'yolo':
            h, w = 1, 1  # 默认值（不读取实际图片尺寸）
            label_lines = []
            for b in boxes:
                cls = class_map.get(b['label'], -1)
                if cls < 0: continue
                cx = (b['x'] + b['width'] / 2) / w
                cy = (b['y'] + b['height'] / 2) / h
                bw = b['width'] / w
                bh = b['height'] / h
                label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            with open(os.path.join(export_path, imgname + '.txt'), 'w') as f:
                f.write('\n'.join(label_lines))

        elif fmt == 'coco':
            # TODO: 可扩展为 COCO JSON（此处先占位）
            pass
        elif fmt == 'voc':
            # TODO: 可扩展为 VOC XML（此处先占位）
            pass

    # 打包为 ZIP
    zipname = f'{export_path}.zip'
    with zipfile.ZipFile(zipname, 'w') as zipf:
        for fname in os.listdir(export_path):
            zipf.write(os.path.join(export_path, fname), arcname=fname)

    shutil.rmtree(export_path)
    return send_file(zipname, as_attachment=True)
if __name__ == '__main__':
    app.run(debug=True)
