# 3D NC Viewer Docker 部署指南

本文档介绍如何使用 Docker 构建和运行 3D NetCDF 文件可视化工具。

## 前置要求

- Docker Engine 20.10+
- Docker Compose 2.0+
- 至少 4GB 可用内存
- 至少 2GB 可用磁盘空间

## 快速开始

### 方法一：使用 Docker Compose（推荐）

1. **克隆或下载项目到本地**

2. **创建数据目录**
   ```bash
   mkdir -p data sample_data
   ```

3. **启动服务**
   ```bash
   docker-compose up -d
   ```

4. **访问应用**
   打开浏览器访问：http://localhost:5000

5. **停止服务**
   ```bash
   docker-compose down
   ```

### 方法二：使用 Docker 命令

1. **构建镜像**
   ```bash
   docker build -t 3dncviewer .
   ```

2. **运行容器**
   ```bash
   docker run -d \
     --name 3dncviewer \
     -p 5000:5000 \
     -v $(pwd)/data:/app/uploads \
     3dncviewer
   ```

3. **访问应用**
   打开浏览器访问：http://localhost:5000

## 配置说明

### 端口配置
- 默认端口：5000
- 可通过修改 `docker-compose.yml` 中的端口映射来更改

### 数据持久化
- 上传的 NC 文件存储在 `./data` 目录
- 示例数据可放在 `./sample_data` 目录

### 环境变量
- `FLASK_ENV`: Flask 运行环境（production/development）
- `PYTHONPATH`: Python 路径设置

## 故障排除

### 常见问题

1. **容器启动失败**
   ```bash
   # 查看日志
   docker-compose logs 3dncviewer
   ```

2. **端口被占用**
   ```bash
   # 修改 docker-compose.yml 中的端口映射
   ports:
     - "8080:5000"  # 改为使用8080端口
   ```

3. **内存不足**
   - 确保系统有足够的可用内存
   - 可以通过 `docker stats` 监控资源使用情况

4. **文件上传失败**
   - 检查 `./data` 目录权限
   - 确保有足够的磁盘空间

### 健康检查

容器包含健康检查功能，可以通过以下命令查看状态：
```bash
docker ps
# 或
docker-compose ps
```

### 日志查看

```bash
# 查看实时日志
docker-compose logs -f 3dncviewer

# 查看最近的日志
docker-compose logs --tail=100 3dncviewer
```

## 生产环境部署

### 使用 Nginx 反向代理

1. 取消注释 `docker-compose.yml` 中的 nginx 服务
2. 创建 `nginx.conf` 配置文件
3. 配置 SSL 证书（如需要）

### 性能优化

1. **资源限制**
   ```yaml
   services:
     3dncviewer:
       deploy:
         resources:
           limits:
             memory: 2G
             cpus: '1.0'
   ```

2. **数据卷优化**
   - 使用 SSD 存储
   - 定期清理临时文件

## 开发环境

如需在开发环境中使用 Docker：

```bash
# 构建开发镜像
docker build -t 3dncviewer:dev .

# 运行开发容器（挂载源代码）
docker run -d \
  --name 3dncviewer-dev \
  -p 5000:5000 \
  -v $(pwd):/app \
  -e FLASK_ENV=development \
  3dncviewer:dev
```

## 更新和维护

### 更新应用
```bash
# 停止服务
docker-compose down

# 重新构建镜像
docker-compose build --no-cache

# 启动服务
docker-compose up -d
```

### 清理资源
```bash
# 清理未使用的镜像
docker image prune

# 清理未使用的容器
docker container prune

# 清理未使用的卷
docker volume prune
```

## 支持

如遇到问题，请检查：
1. Docker 和 Docker Compose 版本
2. 系统资源使用情况
3. 网络连接状态
4. 日志文件中的错误信息