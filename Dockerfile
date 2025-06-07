# 多阶段构建Dockerfile for 3D NC Viewer
# 第一阶段：构建前端依赖
FROM node:18-alpine AS frontend-builder

WORKDIR /app

# 复制package.json和package-lock.json
COPY package*.json ./

# 安装Node.js依赖
RUN npm ci --only=production

# 第二阶段：Python运行时环境
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libnetcdf-dev \
    pkg-config \
    gcc \
    g++ \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# 复制Python依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 从前端构建阶段复制node_modules
COPY --from=frontend-builder /app/node_modules ./node_modules

# 复制应用程序文件
COPY . .

# 确保cesium目录存在并有正确权限
RUN chmod -R 755 cesium/ || true

# 创建上传目录
RUN mkdir -p /app/uploads && chmod 755 /app/uploads

# 暴露端口
EXPOSE 5000

# 设置环境变量
ENV FLASK_APP=backend.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# 启动命令
CMD ["python", "backend.py"]