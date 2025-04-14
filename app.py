# Flask 应用主入口
from flask import Flask, request, render_template, jsonify, redirect, url_for, send_from_directory
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

if __name__ == '__main__':
    app.run(debug=True)
