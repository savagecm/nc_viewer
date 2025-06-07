# 3D NC文件可视化器

一个基于Cesium的3D NetCDF文件可视化工具，可以在三维地球上展示科学数据。

## 🌟 功能特性

- **NC文件读取**: 支持标准NetCDF格式文件的读取和解析
- **3D可视化**: 基于Cesium的三维地球展示
- **交互操作**: 支持拖拽、缩放、旋转等3D交互
- **变量选择**: 可选择NC文件中的不同变量进行可视化
- **颜色映射**: 可调整颜色范围和透明度
- **离线部署**: 完全本地运行，无需网络连接
- **跨平台**: 支持Windows、macOS、Linux

## 📋 系统要求

### Docker部署
- Docker Engine 20.10+
- Docker Compose 2.0+
- 至少4GB可用内存
- 至少2GB可用磁盘空间

### 本地部署
- Python 3.9+
- Node.js 16+（可选）
- 现代浏览器（Chrome、Firefox、Safari、Edge）
- 至少2GB内存（处理大型NC文件时需要更多）

## 🚀 快速开始

### 方法一：Docker部署（推荐）

#### 使用Docker Compose

```bash
# 1. 下载项目
git clone <repository-url>
cd 3dncviewer

# 2. 创建数据目录
mkdir -p data sample_data

# 3. 启动服务
docker-compose up -d

# 4. 访问应用
# 浏览器打开 http://localhost:5000
```

#### 使用Docker命令

```bash
# 1. 构建镜像
docker build -t 3dncviewer .

# 2. 运行容器
docker run -d \
  --name 3dncviewer \
  -p 5000:5000 \
  -v $(pwd)/data:/app/uploads \
  3dncviewer

# 3. 访问应用
# 浏览器打开 http://localhost:5000
```

### 方法二：本地部署

#### 1. 下载项目

```bash
# 如果使用Git
git clone <repository-url>
cd 3dncviewer

# 或者直接下载所有文件到同一目录
```

#### 2. 安装依赖

```bash
# 安装Python依赖
pip install -r requirements.txt

# 安装Node.js依赖（可选）
npm install
```

#### 3. 启动服务器

```bash
# 方法1: 使用Flask后端（推荐）
python3 backend.py

# 方法2: 使用Python内置服务器
python3 -m http.server 5000

# 方法3: 使用Node.js（如果已安装）
npx http-server -p 5000
```

#### 4. 打开浏览器

访问 `http://localhost:5000`

## 📖 使用说明

### 基本操作流程

1. **加载NC文件**
   - 点击"选择文件"按钮
   - 选择.nc格式的文件
   - 点击"加载文件"按钮

2. **查看文件信息**
   - 文件加载后，右侧面板会显示文件的维度、变量和属性信息

3. **选择变量**
   - 在"变量选择"下拉框中选择要可视化的变量
   - 点击"可视化"按钮

4. **调整显示效果**
   - 调整最小值和最大值来改变颜色映射范围
   - 使用透明度滑块调整数据点的透明度
   - 点击"更新颜色"应用更改

### 3D交互操作

- **旋转**: 左键拖拽
- **缩放**: 鼠标滚轮或右键拖拽
- **平移**: 中键拖拽或Shift+左键拖拽
- **倾斜**: Ctrl+左键拖拽

### 支持的NC文件格式

- 标准NetCDF格式（.nc文件）
- 包含经纬度坐标信息
- 支持多维数据数组
- 常见的坐标变量名：
  - 纬度: `lat`, `latitude`, `y`, `LAT`, `LATITUDE`
  - 经度: `lon`, `longitude`, `x`, `LON`, `LONGITUDE`

## 🛠️ 技术架构

### 前端技术栈

- **Cesium.js**: 3D地球可视化引擎
- **HTML5/CSS3/JavaScript**: 用户界面

### 后端技术栈

- **Flask**: Python Web框架
- **netCDF4**: NetCDF文件处理库
- **NumPy**: 科学计算库
- **OpenCV**: 图像处理库

### 核心组件

- `index.html`: 主页面结构
- `app.js`: 前端应用逻辑
- `backend.py`: Flask后端服务
- `styles.css`: 界面样式
- `Dockerfile`: Docker镜像构建文件
- `docker-compose.yml`: 容器编排配置

### 数据处理流程

1. **文件读取**: 使用FileReader API读取本地NC文件
2. **数据解析**: netcdfjs库解析NetCDF格式
3. **坐标提取**: 自动识别经纬度变量
4. **数据映射**: 将数据值映射到3D坐标和颜色
5. **渲染显示**: Cesium引擎渲染3D场景

## 🎨 自定义配置

### 修改服务器端口

#### Docker部署
编辑 `docker-compose.yml` 文件：

```yaml
services:
  3dncviewer:
    ports:
      - "8080:5000"  # 改为你想要的端口
```

#### 本地部署
编辑 `backend.py` 文件中的端口设置：

```python
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)  # 改为你想要的端口
```

### 自定义颜色方案

在 `app.js` 中的 `getColorForValue` 函数中修改颜色映射逻辑：

```javascript
// 当前使用彩虹色谱，可以改为其他颜色方案
const hue = (1 - clamped) * 240; // 从蓝色到红色
return Cesium.Color.fromHsl(hue / 360, 1.0, 0.5);
```

### 调整数据点大小

在 `create3DVisualization` 函数中修改：

```javascript
point: {
    pixelSize: 5, // 改变点的大小
    // ...
}
```

## 🐛 常见问题

### Docker相关问题

### Q: Docker容器启动失败
**A**: 检查Docker服务是否运行，查看容器日志：`docker-compose logs 3dncviewer`

### Q: 端口被占用
**A**: 修改 `docker-compose.yml` 中的端口映射，如改为 `"8080:5000"`

### Q: 文件上传失败
**A**: 确保 `./data` 目录存在且有正确权限，检查磁盘空间是否充足

### 应用相关问题

### Q: 文件加载失败
**A**: 确保文件是标准的NetCDF格式，且包含经纬度坐标信息。

### Q: 可视化后看不到数据
**A**: 检查数据值范围，可能需要调整最小值和最大值设置。

### Q: 浏览器显示CORS错误
**A**: 必须通过HTTP服务器访问，不能直接打开HTML文件。

### Q: 大文件加载缓慢
**A**: 大型NC文件需要更多时间处理，建议使用较小的测试文件。

### Q: 3D场景卡顿
**A**: 减少数据点数量或降低透明度可以提高性能。

## 📝 开发说明

### Docker开发环境

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

### 生产环境部署

详细的生产环境部署指南请参考 [Docker-README.md](Docker-README.md)

### 添加新功能

1. **时间序列支持**: 可以扩展支持时间维度的数据
2. **更多可视化类型**: 添加等值线、热力图等
3. **数据导出**: 支持导出可视化结果
4. **批量处理**: 支持同时加载多个文件

### 性能优化

- 对大型数据集进行采样
- 实现数据分块加载
- 添加WebGL优化
- 使用Web Workers处理数据

### 维护和更新

```bash
# 更新Docker镜像
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# 清理未使用的资源
docker system prune
```

## 📄 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📞 支持

如果遇到问题或有建议，请创建Issue或联系开发者。

---

**享受3D数据可视化的乐趣！** 🌍✨