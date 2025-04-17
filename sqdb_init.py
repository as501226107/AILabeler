import sqlite3
from werkzeug.security import generate_password_hash

# 确保数据库目录存在
import os
os.makedirs("data", exist_ok=True)

# 连接数据库文件
conn = sqlite3.connect("data/ailabeler.db")
cursor = conn.cursor()

# 创建 users 表
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT DEFAULT 'admin'
)
''')

# 检查是否已存在 admin 用户
cursor.execute("SELECT * FROM users WHERE username = 'admin'")
if cursor.fetchone():
    print("管理员用户已存在")
else:
    password_hash = generate_password_hash("admin123")
    cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                   ('admin', password_hash, 'admin'))
    conn.commit()
    print("✅ 管理员账号创建成功：admin / admin123")

conn.close()
