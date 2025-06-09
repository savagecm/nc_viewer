#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D NC文件可视化器 - Flask后端服务
处理NC文件上传、解析和数据提供
"""

from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from flask_cors import CORS
import matplotlib
import netCDF4 as nc
import numpy as np
import os
import tempfile
import json
from werkzeug.utils import secure_filename
import traceback
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import time

# 全局缓存变量
_earth_texture_cache = None
_earth_texture_path_cache = None
import cv2
import matplotlib.cm as cm
from matplotlib.colors import Normalize

app = Flask(__name__)
CORS(app)  # 启用CORS支持

# 配置
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5GB max file size
UPLOAD_FOLDER = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 全局变量存储当前NC数据
current_nc_data = None
current_file_path = None

def allowed_file(filename):
    """检查文件扩展名"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'nc'

@app.route('/')
def index():
    """主页面"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    """静态文件服务"""
    # 排除API路径
    if filename.startswith('api/'):
        return jsonify({'error': 'API endpoint not found'}), 404
    return send_from_directory('.', filename)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """上传NC文件"""
    global current_nc_data, current_file_path
    
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '没有选择文件'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '没有选择文件'})
        
        if not file.filename.lower().endswith('.nc'):
            return jsonify({'success': False, 'error': '只支持.nc格式文件'})
        
        # 保存文件（流式保存以处理大文件）
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        print(f"开始保存大文件: {filename}")
        
        # 分块保存文件以节省内存
        chunk_size = 8192  # 8KB chunks
        with open(filepath, 'wb') as f:
            while True:
                chunk = file.stream.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
        
        print(f"文件保存完成: {filepath}")
        
        # 解析NC文件
        current_file_path = filepath
        
        print("开始解析NC文件...")
        current_nc_data = nc.Dataset(filepath, 'r')
        print("NC文件解析完成")
        
        # 获取文件信息
        file_info = get_file_info(current_nc_data)
        
        return jsonify({
            'success': True,
            'message': '文件上传成功',
            'filename': filename,
            'file_info': file_info
        })
        
    except Exception as e:
        print(f"上传文件错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'上传失败: {str(e)}'}), 500

def get_file_info(dataset):
    """获取NC文件信息"""
    info = {
        'dimensions': {},
        'variables': [],
        'global_attributes': {}
    }
    
    # 维度信息
    for dim_name, dim in dataset.dimensions.items():
        info['dimensions'][dim_name] = {
            'name': dim_name,
            'size': len(dim)
        }
    
    # 变量信息
    for var_name, var in dataset.variables.items():
        var_info = {
            'name': var_name,
            'dimensions': list(var.dimensions),
            'shape': list(var.shape),
            'dtype': str(var.dtype)
        }
        
        # 添加变量属性
        attributes = {}
        for attr_name in var.ncattrs():
            try:
                attr_value = getattr(var, attr_name)
                if isinstance(attr_value, np.ndarray):
                    attr_value = attr_value.tolist()
                elif isinstance(attr_value, (np.integer, np.floating)):
                    attr_value = float(attr_value)
                    # 检查是否为NaN或无穷大
                    if np.isnan(attr_value) or np.isinf(attr_value):
                        attr_value = str(attr_value)
                elif isinstance(attr_value, bytes):
                    attr_value = attr_value.decode('utf-8', errors='ignore')
                attributes[attr_name] = attr_value
            except Exception as e:
                attributes[attr_name] = str(getattr(var, attr_name))
        
        var_info['attributes'] = attributes
        info['variables'].append(var_info)
    
    # 全局属性
    for attr_name in dataset.ncattrs():
        try:
            attr_value = getattr(dataset, attr_name)
            if isinstance(attr_value, np.ndarray):
                attr_value = attr_value.tolist()
            elif isinstance(attr_value, (np.integer, np.floating)):
                attr_value = float(attr_value)
                # 检查是否为NaN或无穷大
                if np.isnan(attr_value) or np.isinf(attr_value):
                    attr_value = str(attr_value)
            elif isinstance(attr_value, bytes):
                attr_value = attr_value.decode('utf-8', errors='ignore')
            info['global_attributes'][attr_name] = attr_value
        except Exception as e:
            info['global_attributes'][attr_name] = str(getattr(dataset, attr_name))
    
    return info

@app.route('/api/variables', methods=['GET'])
def get_variables():
    """获取变量列表"""
    global current_nc_data
    
    if current_nc_data is None:
        return jsonify({'error': '没有加载NC文件'}), 400
    
    try:
        variables = []
        for var_name, var in current_nc_data.variables.items():
            # 只返回多维数据变量
            if len(var.dimensions) >= 2:
                # 获取维度详细信息
                dimension_info = []
                for i, dim_name in enumerate(var.dimensions):
                    dim_size = var.shape[i]
                    dimension_info.append({
                        'name': dim_name,
                        'size': dim_size,
                        'index': i
                    })
                
                variables.append({
                    'name': var_name,
                    'dimensions': list(var.dimensions),
                    'shape': list(var.shape),
                    'dimension_info': dimension_info,
                    'long_name': getattr(var, 'long_name', var_name),
                    'units': getattr(var, 'units', '')
                })
        
        return jsonify({'variables': variables})
        
    except Exception as e:
        print(f"获取变量列表错误: {e}")
        return jsonify({'error': f'获取变量失败: {str(e)}'}), 500

@app.route('/api/coordinates', methods=['GET'])
def get_coordinates():
    """获取坐标信息"""
    global current_nc_data
    
    if current_nc_data is None:
        return jsonify({'error': '没有加载NC文件'}), 400
    
    try:
        # 常见的坐标变量名
        lat_names = ['lat', 'latitude', 'y', 'LAT', 'LATITUDE', 'Latitude']
        lon_names = ['lon', 'longitude', 'x', 'LON', 'LONGITUDE', 'Longitude']
        
        latitudes = None
        longitudes = None
        lat_var_name = None
        lon_var_name = None
        
        # 查找纬度
        for name in lat_names:
            if name in current_nc_data.variables:
                lat_data = current_nc_data.variables[name][:]
                # 处理NaN值
                lat_clean = np.where(np.isnan(lat_data) | np.isinf(lat_data), None, lat_data)
                latitudes = lat_clean.tolist()
                lat_var_name = name
                break
        
        # 查找经度
        for name in lon_names:
            if name in current_nc_data.variables:
                lon_data = current_nc_data.variables[name][:]
                # 处理NaN值
                lon_clean = np.where(np.isnan(lon_data) | np.isinf(lon_data), None, lon_data)
                longitudes = lon_clean.tolist()
                lon_var_name = name
                break
        
        if latitudes is None or longitudes is None:
            return jsonify({'error': '找不到经纬度坐标变量'}), 400
        
        return jsonify({
            'latitudes': latitudes,
            'longitudes': longitudes,
            'lat_var_name': lat_var_name,
            'lon_var_name': lon_var_name
        })
        
    except Exception as e:
        print(f"获取坐标错误: {e}")
        return jsonify({'error': f'获取坐标失败: {str(e)}'}), 500

def extract_coordinates(dataset):
    """提取坐标信息，返回字典格式"""
    try:
        # 常见的坐标变量名
        lat_names = ['lat', 'latitude', 'y', 'LAT', 'LATITUDE', 'Latitude']
        lon_names = ['lon', 'longitude', 'x', 'LON', 'LONGITUDE', 'Longitude']
        
        latitudes = None
        longitudes = None
        lat_var_name = None
        lon_var_name = None
        
        # 查找纬度
        for name in lat_names:
            if name in dataset.variables:
                lat_data = dataset.variables[name][:]
                # 处理NaN值
                lat_clean = np.where(np.isnan(lat_data) | np.isinf(lat_data), None, lat_data)
                latitudes = lat_clean.tolist()
                lat_var_name = name
                break
        
        # 查找经度
        for name in lon_names:
            if name in dataset.variables:
                lon_data = dataset.variables[name][:]
                # 处理NaN值
                lon_clean = np.where(np.isnan(lon_data) | np.isinf(lon_data), None, lon_data)
                longitudes = lon_clean.tolist()
                lon_var_name = name
                break
        
        if latitudes is None or longitudes is None:
            return None
        
        return {
            'latitudes': latitudes,
            'longitudes': longitudes,
            'lat_var_name': lat_var_name,
            'lon_var_name': lon_var_name
        }
        
    except Exception as e:
        print(f"提取坐标错误: {e}")
        return None

@app.route('/api/image/<variable_name>', methods=['GET'])
def get_variable_image(variable_name):
    """生成变量数据的灰度图像并保存为PNG文件"""
    global current_nc_data
    
    try:
        if not current_nc_data:
            return jsonify({'error': '没有打开的NC文件'}), 400
        
        if variable_name not in current_nc_data.variables:
            return jsonify({'error': f'变量 {variable_name} 不存在'}), 400
        
        # 获取维度选择参数
        dimension_indices = {}
        for key, value in request.args.items():
            if key.startswith('dim_'):
                dim_name = key[4:]  # 移除 'dim_' 前缀
                try:
                    dimension_indices[dim_name] = int(value)
                except ValueError:
                    return jsonify({'error': f'维度索引必须是整数: {key}={value}'}), 400
        
        # 获取颜色方案参数
        color_scheme = request.args.get('colorScheme', 'viridis')
        # 获取是否生成灰度图的参数
        grayscale = request.args.get('grayscale', 'false').lower() == 'true'
        print(f"生成图像: {variable_name}，颜色方案: {color_scheme}，灰度模式: {grayscale}")
        
        print(f"开始生成变量图像: {variable_name}, 维度选择: {dimension_indices}")
        
        var = current_nc_data.variables[variable_name]
        data = var[:]
        
        # 获取经纬度信息
        lat_var = None
        lon_var = None
        
        # 查找经纬度变量
        for var_name in current_nc_data.variables:
            var_obj = current_nc_data.variables[var_name]
            if hasattr(var_obj, 'standard_name'):
                if var_obj.standard_name in ['latitude', 'lat']:
                    lat_var = var_obj
                elif var_obj.standard_name in ['longitude', 'lon']:
                    lon_var = var_obj
            elif var_name.lower() in ['lat', 'latitude', 'y']:
                lat_var = var_obj
            elif var_name.lower() in ['lon', 'longitude', 'x']:
                lon_var = var_obj
        
        if lat_var is None or lon_var is None:
            # 如果找不到经纬度变量，尝试从维度中获取
            dims = var.dimensions
            for dim_name in dims:
                if dim_name.lower() in ['lat', 'latitude', 'y'] and dim_name in current_nc_data.variables:
                    lat_var = current_nc_data.variables[dim_name]
                elif dim_name.lower() in ['lon', 'longitude', 'x'] and dim_name in current_nc_data.variables:
                    lon_var = current_nc_data.variables[dim_name]
        
        # 处理多维数据，根据用户选择的维度进行切片
        original_shape = data.shape
        if len(data.shape) > 2:
            print(f"多维数据，原始形状: {original_shape}")
            
            # 构建切片索引
            slice_indices = [slice(None)] * len(data.shape)
            
            # 应用用户选择的维度索引
            for dim_name, dim_index in dimension_indices.items():
                if dim_name in var.dimensions:
                    dim_position = list(var.dimensions).index(dim_name)
                    if 0 <= dim_index < data.shape[dim_position]:
                        slice_indices[dim_position] = dim_index
                    else:
                        return jsonify({'error': f'维度 {dim_name} 的索引 {dim_index} 超出范围 [0, {data.shape[dim_position]-1}]'}), 400
            
            # 如果没有指定维度选择，使用默认策略
            if not dimension_indices:
                # 找到最大的2D切片
                if len(data.shape) == 3:
                    slice_indices[0] = 0
                elif len(data.shape) == 4:
                    slice_indices[0] = 0
                    slice_indices[1] = 0
                else:
                    # 对于更高维度，将前面的维度设为0
                    for i in range(len(data.shape) - 2):
                        slice_indices[i] = 0
            
            # 应用切片
            data = data[tuple(slice_indices)]
            
            # 确保结果是2D的
            while len(data.shape) > 2:
                data = np.squeeze(data, axis=0)
        
        print(f"处理后数据形状: {data.shape}")
        
        # 处理masked array
        if hasattr(data, 'mask'):
            data = np.ma.filled(data, np.nan)
        
        # 移除NaN值
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            return jsonify({'error': '数据中没有有效值'}), 400
        
        # 数据归一化到0-255
        valid_data = data[valid_mask]
        data_min = np.min(valid_data)
        data_max = np.max(valid_data)
        
        if data_max == data_min:
            normalized_data = np.zeros_like(data)
        else:
            normalized_data = (data - data_min) / (data_max - data_min)
        
        # 将NaN设置为0（黑色）
        normalized_data[~valid_mask] = 0
        
        # 自定义颜色映射函数
        def apply_custom_colormap(data, colormap):
            """应用自定义颜色映射到归一化数据"""
            height, width = data.shape
            rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 将颜色映射转换为numpy数组以便向量化操作
            positions = np.array([item['position'] for item in colormap])
            colors = np.array([item['color'] for item in colormap])
            
            # 对每个像素进行颜色插值
            for i in range(height):
                for j in range(width):
                    value = data[i, j]
                    if np.isnan(value):
                        rgb_image[i, j] = [0, 0, 0]  # NaN值设为黑色
                        continue
                    
                    # 找到对应的颜色区间
                    if value <= positions[0]:
                        rgb_image[i, j] = colors[0]
                    elif value >= positions[-1]:
                        rgb_image[i, j] = colors[-1]
                    else:
                        # 线性插值
                        for k in range(len(positions) - 1):
                            if positions[k] <= value <= positions[k + 1]:
                                t = (value - positions[k]) / (positions[k + 1] - positions[k])
                                rgb_image[i, j] = colors[k] + t * (colors[k + 1] - colors[k])
                                break
            
            return rgb_image.astype(np.uint8)
        
        # 应用颜色映射（优化版本，使用向量化操作）
        def apply_colormap(data, colormap_name, custom_colormap=None):
            """应用颜色映射到归一化数据 - 支持matplotlib内置颜色映射和自定义颜色映射"""
            import matplotlib.cm as cm
            
            # 如果使用自定义颜色映射
            if custom_colormap and colormap_name == 'custom':
                return apply_custom_colormap(data, custom_colormap)
            
            # 颜色方案映射
            color_scheme_mapping = {
                'grayscale': 'gray',
                'grey': 'gray',
                'greys': 'Greys'
            }
            
            # 映射颜色方案名称
            mapped_scheme = color_scheme_mapping.get(colormap_name, colormap_name)
            
            # 获取matplotlib颜色映射
            try:
                cmap = cm.get_cmap(mapped_scheme)
            except ValueError:
                print(f"警告: 颜色方案 '{colormap_name}' 不支持，使用默认的 'viridis'")
                cmap = cm.get_cmap('viridis')
            
            # 生成颜色数组
            colors = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
            height, width = data.shape
            
            def process_chunk(start_row, end_row, data_chunk, valid_mask_chunk, colors):
                """处理数据块的函数"""
                chunk_height = end_row - start_row
                rgb_chunk = np.zeros((chunk_height, width, 3), dtype=np.uint8)
                
                # 向量化处理有效数据
                valid_data = data_chunk[valid_mask_chunk]
                if len(valid_data) > 0:
                    # 计算颜色索引
                    color_indices = valid_data * (len(colors) - 1)
                    indices = np.floor(color_indices).astype(np.int32)
                    t_values = color_indices - indices
                    
                    # 确保索引在有效范围内
                    indices = np.clip(indices, 0, len(colors) - 2)
                    
                    # 获取颜色对
                    color1 = colors[indices]
                    color2 = colors[indices + 1]
                    
                    # 线性插值
                    interpolated_colors = color1 + t_values[:, np.newaxis] * (color2 - color1)
                    
                    # 将结果放回对应位置
                    rgb_chunk[valid_mask_chunk] = interpolated_colors.astype(np.uint8)
                
                return rgb_chunk
            
            # 如果数据较小，直接处理
            if height * width < 100000:  # 小于100k像素直接处理
                return process_chunk(0, height, data, valid_mask, colors)
            
            # 大数据使用多线程处理
            num_threads = min(4, height // 100)  # 最多4个线程，每个线程至少处理100行
            if num_threads <= 1:
                return process_chunk(0, height, data, valid_mask, colors)
            
            chunk_size = height // num_threads
            rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for i in range(num_threads):
                    start_row = i * chunk_size
                    end_row = height if i == num_threads - 1 else (i + 1) * chunk_size
                    
                    data_chunk = data[start_row:end_row]
                    valid_mask_chunk = valid_mask[start_row:end_row]
                    
                    future = executor.submit(process_chunk, start_row, end_row, data_chunk, valid_mask_chunk, colors)
                    futures.append((future, start_row, end_row))
                
                # 收集结果
                for future, start_row, end_row in futures:
                    rgb_chunk = future.result()
                    rgb_image[start_row:end_row] = rgb_chunk
            
            return rgb_image
        
        # 根据模式生成图像（优化版本）
        if grayscale:
            # 生成灰度图（向量化优化）
            print(f"开始生成灰度图，数据尺寸: {normalized_data.shape}")
            start_time = time.time()
            
            # 向量化操作：直接对整个数组进行处理
            gray_data = (normalized_data * 255).astype(np.uint8)
            
            # 创建RGBA图像以支持透明度
            height, width = gray_data.shape
            rgba_data = np.zeros((height, width, 4), dtype=np.uint8)
            
            # 设置RGB通道为灰度值
            rgba_data[:, :, 0] = gray_data  # R
            rgba_data[:, :, 1] = gray_data  # G
            rgba_data[:, :, 2] = gray_data  # B
            
            # 设置Alpha通道：有效数据为不透明，无效数据为透明
            rgba_data[:, :, 3] = np.where(valid_mask, 255, 0)  # Alpha
            
            # 翻转Y轴（图像坐标系与地理坐标系相反）
            rgba_data = np.flipud(rgba_data)
            
            # 创建PIL RGBA图像
            img = Image.fromarray(rgba_data, mode='RGBA')
            
            end_time = time.time()
            print(f"灰度图生成完成，耗时: {end_time - start_time:.3f}秒")
        else:
            # 应用颜色映射生成彩色图（多线程优化）
            print(f"开始生成彩色图，数据尺寸: {normalized_data.shape}，颜色方案: {color_scheme}，透明度: {opacity}")
            start_time = time.time()
            
            rgb_data = apply_colormap(normalized_data, color_scheme, custom_colormap)
            
            # 翻转Y轴（图像坐标系与地理坐标系相反）
            rgb_data = np.flipud(rgb_data)
            
            # 如果透明度不是1.0，创建RGBA图像
            if opacity < 1.0:
                height, width = rgb_data.shape[:2]
                rgba_data = np.zeros((height, width, 4), dtype=np.uint8)
                rgba_data[:, :, :3] = rgb_data  # RGB通道
                rgba_data[:, :, 3] = np.where(valid_mask, int(opacity * 255), 0)  # Alpha通道
                rgba_data[:, :, 3] = np.flipud(rgba_data[:, :, 3])  # 翻转Alpha通道
                img = Image.fromarray(rgba_data, mode='RGBA')
            else:
                # 创建PIL图像
                img = Image.fromarray(rgb_data, mode='RGB')
            
            end_time = time.time()
            print(f"彩色图生成完成，耗时: {end_time - start_time:.3f}秒")
        
        # 保存图像到文件系统
        import os
        images_dir = os.path.join(os.path.dirname(__file__), 'generated_images')
        os.makedirs(images_dir, exist_ok=True)
        
        image_filename = f'{variable_name}.png'
        image_path = os.path.join(images_dir, image_filename)
        img.save(image_path, 'PNG')
        
        print(f"图像已保存到: {image_path}，尺寸: {img.size}，颜色方案: {color_scheme}")
        
        # 返回JSON格式，包含图像URL和数据范围信息
        return jsonify({
            'success': True,
            'image_url': f'/api/images/{image_filename}',
            'data_min': float(data_min),
            'data_max': float(data_max),
            'image_size': {
                'width': img.size[0],
                'height': img.size[1]
            },
            'variable_name': variable_name
        })
        
    except Exception as e:
        print(f"生成图像错误: {e}")
        print(traceback.format_exc())
        return jsonify({'error': f'生成图像失败: {str(e)}'}), 500

@app.route('/api/images/<filename>', methods=['GET'])
def serve_generated_image(filename):
    """提供生成的图像文件"""
    try:
        import os
        images_dir = os.path.join(os.path.dirname(__file__), 'generated_images')
        image_path = os.path.join(images_dir, filename)
        
        if os.path.exists(image_path):
            return send_file(
                image_path,
                mimetype='image/png',
                as_attachment=False
            )
        else:
            return jsonify({'error': '图像文件不存在'}), 404
            
    except Exception as e:
        print(f"提供图像文件错误: {e}")
        return jsonify({'error': f'获取图像失败: {str(e)}'}), 500

@app.route('/api/visualization/<variable_name>', methods=['GET'])
def get_visualization_data(variable_name):
    """获取变量的可视化数据，支持分页和瓦片化传输"""
    global current_nc_data
    
    if not current_nc_data:
        return jsonify({'error': '没有打开的NC文件'}), 400
    
    try:
        if variable_name not in current_nc_data.variables:
            return jsonify({'error': f'变量 {variable_name} 不存在'}), 400
        
        # 获取参数
        color_scheme = request.args.get('color_scheme', 'viridis')
        custom_colormap_str = request.args.get('custom_colormap')
        custom_colormap = None
        
        # 解析自定义颜色映射
        if custom_colormap_str:
            try:
                custom_colormap = json.loads(custom_colormap_str)
                print(f"使用自定义颜色映射，包含 {len(custom_colormap)} 个颜色点")
            except json.JSONDecodeError:
                print("自定义颜色映射解析失败，使用默认颜色方案")
                custom_colormap = None
        opacity = float(request.args.get('opacity', 1.0))
        
        # 分页参数
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 50000))  # 默认每页5万个点
        
        # 空间范围参数（瓦片化）
        lat_min = request.args.get('lat_min', type=float)
        lat_max = request.args.get('lat_max', type=float)
        lon_min = request.args.get('lon_min', type=float)
        lon_max = request.args.get('lon_max', type=float)
        
        # 采样参数
        sample_rate = float(request.args.get('sample_rate', 1.0))  # 采样率，1.0表示全部数据
        
        # 获取维度参数
        time_index = request.args.get('time_index', type=int)
        depth_index = request.args.get('depth_index', type=int)
        level_index = request.args.get('level', type=int)
        # 支持im_level参数（与level参数等效）
        if level_index is None:
            level_index = request.args.get('im_level', type=int)
        
        print(f"开始处理变量 {variable_name} 的可视化数据")
        print(f"颜色方案: {color_scheme}")
        print(f"分页参数: 第{page}页, 每页{page_size}个点")
        print(f"采样率: {sample_rate}")
        if lat_min is not None:
            print(f"空间范围: 纬度[{lat_min}, {lat_max}], 经度[{lon_min}, {lon_max}]")
        
        # 获取变量数据
        var_data = current_nc_data.variables[variable_name]
        print(f"变量形状: {var_data.shape}")
        
        # 获取坐标信息
        coordinates = extract_coordinates(current_nc_data)
        if not coordinates:
            return jsonify({'error': '无法提取坐标信息'}), 400
        
        latitudes = np.array(coordinates['latitudes'])
        longitudes = np.array(coordinates['longitudes'])
        
        # 空间范围过滤
        if lat_min is not None and lat_max is not None:
            lat_indices = np.where((latitudes >= lat_min) & (latitudes <= lat_max))[0]
        else:
            lat_indices = np.arange(len(latitudes))
            
        if lon_min is not None and lon_max is not None:
            lon_indices = np.where((longitudes >= lon_min) & (longitudes <= lon_max))[0]
        else:
            lon_indices = np.arange(len(longitudes))
        
        print(f"过滤后纬度范围: {len(lat_indices)} 个点")
        print(f"过滤后经度范围: {len(lon_indices)} 个点")
        
        # 根据维度提取数据
        original_shape = var_data.shape
        
        if len(var_data.shape) == 4:  # (time, depth, lat, lon)
            t_idx = time_index if time_index is not None else 0
            d_idx = depth_index if depth_index is not None else 0
            data = var_data[t_idx, d_idx, :, :]
            print(f"提取4D数据切片 [时间:{t_idx}, 深度:{d_idx}, :, :]")
        elif len(var_data.shape) == 3:
            # 需要判断是 (time, lat, lon) 还是 (depth, lat, lon) 或其他
            if time_index is not None:
                data = var_data[time_index, :, :]
                print(f"提取3D数据切片 [时间:{time_index}, :, :]")
            elif depth_index is not None:
                data = var_data[depth_index, :, :]
                print(f"提取3D数据切片 [深度:{depth_index}, :, :]")
            else:
                data = var_data[0, :, :]
                print("提取3D数据切片 [0, :, :]")
        elif len(var_data.shape) == 2:  # (lat, lon)
            data = var_data[:, :]
            print("使用2D数据")
        else:
            return jsonify({'error': f'不支持的数据维度: {var_data.shape}'}), 400
        
        # 应用空间范围过滤
        # 确保索引是整数数组
        lat_indices = np.asarray(lat_indices, dtype=int)
        lon_indices = np.asarray(lon_indices, dtype=int)
        
        # 使用ix_进行二维索引
        data = data[np.ix_(lat_indices, lon_indices)]
        filtered_latitudes = latitudes[lat_indices]
        filtered_longitudes = longitudes[lon_indices]
        
        # 转换为numpy数组并处理masked array
        data = np.array(data)
        
        # 处理 masked array 和 NaN 值
        if hasattr(data, 'mask'):
            # 如果是 masked array，创建有效数据的掩码
            valid_mask = ~data.mask
            data = data.data  # 获取实际数据
        else:
            valid_mask = np.ones(data.shape, dtype=bool)
        
        # 进一步过滤 NaN、无穷大和异常值
        finite_mask = np.isfinite(data)
        valid_mask = valid_mask & finite_mask
        
        # 使用更严格的异常值过滤方法
        if np.any(valid_mask):
            # 首先应用绝对值上限过滤（10^10）
            absolute_limit = 1e10
            absolute_mask = np.abs(data) <= absolute_limit
            valid_mask = valid_mask & absolute_mask
            
            if np.any(valid_mask):
                valid_data = data[valid_mask]
                
                # 使用百分位数方法移除极端值
                p1 = np.percentile(valid_data, 1)   # 第1百分位数
                p99 = np.percentile(valid_data, 99) # 第99百分位数
                
                # 过滤掉超出1%-99%范围的极端值
                percentile_mask = (data >= p1) & (data <= p99)
                valid_mask = valid_mask & percentile_mask
                
                # 在过滤后的数据上再次应用IQR方法
                if np.any(valid_mask):
                    filtered_data = data[valid_mask]
                    q1 = np.percentile(filtered_data, 25)
                    q3 = np.percentile(filtered_data, 75)
                    iqr = q3 - q1
                    
                    # 使用更严格的1.5倍IQR标准
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    # 创建最终的异常值掩码
                    final_outlier_mask = (data >= lower_bound) & (data <= upper_bound)
                    valid_mask = valid_mask & final_outlier_mask
                    
                    print(f"绝对值过滤: 上限={absolute_limit:.0e}")
                    print(f"百分位数过滤: P1={p1:.6f}, P99={p99:.6f}")
                    print(f"IQR统计: Q1={q1:.6f}, Q3={q3:.6f}, IQR={iqr:.6f}")
                    print(f"最终过滤范围: [{lower_bound:.6f}, {upper_bound:.6f}]")
        
        # 计算有效数据的统计信息
        if np.any(valid_mask):
            valid_data = data[valid_mask]
            data_min = float(np.min(valid_data))
            data_max = float(np.max(valid_data))
            total_valid_count = np.sum(valid_mask)
            print(f"有效数据点数: {total_valid_count}, 数据范围: [{data_min:.6f}, {data_max:.6f}]")
        else:
            return jsonify({'error': '没有有效的数据点'}), 400
        
        # 生成所有有效点的坐标列表
        valid_points = []
        for i in range(len(filtered_latitudes)):
            for j in range(len(filtered_longitudes)):
                if valid_mask[i, j]:
                    valid_points.append({
                        'i': i, 'j': j,
                        'lat': float(filtered_latitudes[i]),
                        'lon': float(filtered_longitudes[j]),
                        'value': float(data[i, j])
                    })
        
        # 应用采样
        if sample_rate < 1.0:
            sample_count = int(len(valid_points) * sample_rate)
            valid_points = np.random.choice(valid_points, size=sample_count, replace=False).tolist()
            print(f"采样后数据点数: {len(valid_points)}")
        
        # 分页处理
        total_points = len(valid_points)
        total_pages = (total_points + page_size - 1) // page_size
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_points)
        
        page_points = valid_points[start_idx:end_idx]
        
        print(f"分页信息: 总点数={total_points}, 总页数={total_pages}, 当前页={page}")
        print(f"当前页范围: [{start_idx}, {end_idx}), 实际点数={len(page_points)}")
        
        # 生成当前页的可视化数据点
        visualization_points = []
        for point in page_points:
            # 归一化值
            normalized_value = (point['value'] - data_min) / (data_max - data_min) if data_max != data_min else 0.5
            
            # 获取颜色
            rgb_color = get_color_for_value(normalized_value, color_scheme, custom_colormap)
            
            visualization_points.append({
                'longitude': point['lon'],
                'latitude': point['lat'],
                'value': point['value'],
                'color': rgb_color
            })
        
        # 生成颜色条数据
        colorbar_data = []
        num_steps = 50
        for i in range(num_steps):
            normalized_value = i / (num_steps - 1)
            value = data_min + normalized_value * (data_max - data_min)
            rgb_color = get_color_for_value(normalized_value, color_scheme, custom_colormap)
            colorbar_data.append({
                'value': value,
                'color': rgb_color,
                'normalized': normalized_value
            })
        
        # 返回分页的可视化数据
        result = {
            'success': True,
            'variable_name': variable_name,
            'page': page,
            'page_size': page_size,
            'total_pages': total_pages,
            'total_points': total_points,
            'current_page_points': len(visualization_points),
            'data_range': {
                'min': data_min,
                'max': data_max
            },
            'color_scheme': color_scheme,
            'points': visualization_points,
            'colorbar': colorbar_data,
            'metadata': {
                'units': getattr(var_data, 'units', ''),
                'long_name': getattr(var_data, 'long_name', variable_name),
                'original_shape': list(original_shape),
                'processed_shape': list(data.shape)
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"获取可视化数据错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'获取可视化数据失败: {str(e)}'}), 500

def get_color_for_value(normalized_value, scheme='viridis', custom_colormap=None):
    """根据归一化值和颜色方案获取RGB颜色 - 支持matplotlib内置颜色映射和自定义颜色映射"""
    
    # 如果使用自定义颜色映射
    if custom_colormap and scheme == 'custom':
        return get_color_from_custom_colormap(normalized_value, custom_colormap)
    
    import matplotlib.cm as cm
    
    # 颜色方案映射
    color_scheme_mapping = {
        'grayscale': 'gray',
        'grey': 'gray',
        'greys': 'Greys'
    }
    
    # 映射颜色方案名称
    mapped_scheme = color_scheme_mapping.get(scheme, scheme)
    
    # 获取matplotlib颜色映射
    try:
        cmap = cm.get_cmap(mapped_scheme)
    except ValueError:
        print(f"警告: 颜色方案 '{scheme}' 不支持，使用默认的 'viridis'")
        cmap = cm.get_cmap('viridis')
    
    # 获取颜色值
    rgba = cmap(normalized_value)
    
    # 转换为RGB格式 (0-255)
    return [int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)]

def get_color_from_custom_colormap(normalized_value, colormap):
    """从自定义颜色映射获取颜色"""
    if not colormap or len(colormap) == 0:
        return [0, 0, 0]
    
    # 如果只有一个颜色点
    if len(colormap) == 1:
        return colormap[0]['color']
    
    # 找到对应的颜色区间
    for i in range(len(colormap) - 1):
        current = colormap[i]
        next_color = colormap[i + 1]
        
        if current['position'] <= normalized_value <= next_color['position']:
            # 线性插值
            t = (normalized_value - current['position']) / (next_color['position'] - current['position'])
            r = int(current['color'][0] + t * (next_color['color'][0] - current['color'][0]))
            g = int(current['color'][1] + t * (next_color['color'][1] - current['color'][1]))
            b = int(current['color'][2] + t * (next_color['color'][2] - current['color'][2]))
            return [r, g, b]
    
    # 如果超出范围，返回最近的颜色
    if normalized_value <= colormap[0]['position']:
        return colormap[0]['color']
    else:
        return colormap[-1]['color']

def apply_custom_colormap_vectorized(normalized_values, colormap):
    """向量化应用自定义颜色映射，大幅提升性能"""
    if not colormap or len(colormap) == 0:
        return np.zeros((len(normalized_values), 3), dtype=np.uint8)
    
    # 如果只有一个颜色点
    if len(colormap) == 1:
        color = np.array(colormap[0]['color'], dtype=np.uint8)
        return np.tile(color, (len(normalized_values), 1))
    
    # 提取位置和颜色数组
    positions = np.array([point['position'] for point in colormap])
    colors = np.array([point['color'] for point in colormap], dtype=np.float32)
    
    # 使用numpy的searchsorted找到插值区间
    indices = np.searchsorted(positions, normalized_values, side='right') - 1
    
    # 处理边界情况
    indices = np.clip(indices, 0, len(colormap) - 2)
    
    # 计算插值权重
    pos_left = positions[indices]
    pos_right = positions[indices + 1]
    
    # 避免除零错误
    pos_diff = pos_right - pos_left
    pos_diff = np.where(pos_diff == 0, 1, pos_diff)
    
    t = (normalized_values - pos_left) / pos_diff
    t = np.clip(t, 0, 1)
    
    # 线性插值颜色
    colors_left = colors[indices]
    colors_right = colors[indices + 1]
    
    interpolated_colors = colors_left + t[:, np.newaxis] * (colors_right - colors_left)
    
    return interpolated_colors.astype(np.uint8)


@app.route('/api/visualization/<variable_name>/image', methods=['GET'])
def generate_visualization_image(variable_name):
    """生成可视化图片，按照三个步骤处理：地球纹理预处理、NC数据处理、图像合成"""
    global current_nc_data
    
    if not current_nc_data:
        return jsonify({'error': '没有打开的NC文件'}), 400
    
    try:
        if variable_name not in current_nc_data.variables:
            return jsonify({'error': f'变量 {variable_name} 不存在'}), 400
        
        # 获取参数
        color_scheme = request.args.get('color_scheme', 'viridis')
        custom_colormap_str = request.args.get('custom_colormap')
        custom_colormap = None
        
        # 解析自定义颜色映射
        if custom_colormap_str:
            try:
                custom_colormap = json.loads(custom_colormap_str)
                print(f"使用自定义颜色映射，包含 {len(custom_colormap)} 个颜色点")
            except json.JSONDecodeError:
                print("自定义颜色映射解析失败，使用默认颜色方案")
                custom_colormap = None
        
        # 获取维度参数
        time_index = request.args.get('time_index', type=int)
        depth_index = request.args.get('depth_index', type=int)
        level_index = request.args.get('level', type=int)
        if level_index is None:
            level_index = request.args.get('im_level', type=int)
        
        print(f"生成变量 {variable_name} 的可视化图片")
        print(f"参数: 颜色方案={color_scheme}")
        print(f"层级参数: time_index={time_index}, depth_index={depth_index}, level_index={level_index}")
        
        # 获取变量数据和坐标信息
        var_data = current_nc_data.variables[variable_name]
        coordinates = extract_coordinates(current_nc_data)
        if not coordinates:
            return jsonify({'error': '无法提取坐标信息'}), 400
        
        latitudes = np.array(coordinates['latitudes'])
        longitudes = np.array(coordinates['longitudes'])
        print(f"var_data.shape is {var_data.shape}")
        # 根据维度提取数据
        if len(var_data.shape) == 4:  # (time, depth/level, lat, lon)
            t_idx = time_index if time_index is not None else 0
            d_idx = level_index if level_index is not None else (depth_index if depth_index is not None else 0)
            data = var_data[t_idx, d_idx, :, :]
            print(f"提取4D数据: time_index={t_idx}, level/depth_index={d_idx}")
        elif len(var_data.shape) == 3:
            if time_index is not None:
                data = var_data[time_index, :, :]
                print(f"提取3D数据: time_index={time_index}")
            elif level_index is not None:
                data = var_data[level_index, :, :]
                print(f"提取3D数据: level_index={level_index}")
            elif depth_index is not None:
                data = var_data[depth_index, :, :]
                print(f"提取3D数据: depth_index={depth_index}")
            else:
                data = var_data[0, :, :]
                print("提取3D数据: 使用默认索引0")
        elif len(var_data.shape) == 2:  # (lat, lon)
            data = var_data[:, :]
            print("提取2D数据")
        else:
            return jsonify({'error': f'不支持的数据维度: {var_data.shape}'}), 400
        
        # 转换为numpy数组并翻转纬度方向
        data = np.array(data)
        data = np.flipud(data)  # NC文件纬度通常从北到南，翻转为从南到北
        print(f"数据已沿纬度方向翻转")
        
        # 获取数据维度和经纬度范围
        data_height, data_width = data.shape  # lat, lon
        latitudes = np.flipud(latitudes)  # 对应数据翻转
        
        lat_min, lat_max = float(np.min(latitudes)), float(np.max(latitudes))
        lon_min, lon_max = float(np.min(longitudes)), float(np.max(longitudes))
        
        # 处理经度范围：统一转换为[-180, 180]格式
        if lon_min >= 0 and lon_max > 180:
            # 将[0, 360]格式转换为[-180, 180]格式
            # 对于经度数据，需要重新排列数据
            longitudes_mapped = np.where(longitudes > 180, longitudes - 360, longitudes)
            
            # 如果数据跨越了180度经线，需要重新排列数据
            if np.any(longitudes > 180):
                # 找到180度的分界点
                split_idx = np.where(longitudes > 180)[0][0]
                
                # 重新排列经度数组
                longitudes = np.concatenate([longitudes[split_idx:] - 360, longitudes[:split_idx]])
                
                # 重新排列数据数组（沿经度维度）
                data = np.concatenate([data[:, split_idx:], data[:, :split_idx]], axis=1)
                
                print(f"数据已重新排列以适应[-180, 180]经度格式")
            
            lon_min_mapped = float(np.min(longitudes))
            lon_max_mapped = float(np.max(longitudes))
            print(f"经度范围从[{lon_min:.2f}, {lon_max:.2f}]转换为[{lon_min_mapped:.2f}, {lon_max_mapped:.2f}]")
        else:
            lon_min_mapped = lon_min
            lon_max_mapped = lon_max
        
        print(f"数据维度: {data_height}x{data_width} (纬度x经度)")
        print(f"数据地理范围: 纬度[{lat_min:.2f}, {lat_max:.2f}], 经度[{lon_min_mapped:.2f}, {lon_max_mapped:.2f}]")
        
        # 处理数据异常值和掩码
        if hasattr(data, 'mask'):
            valid_mask = ~data.mask
            data = data.data
        else:
            valid_mask = np.ones(data.shape, dtype=bool)
        
        # 过滤异常值
        finite_mask = np.isfinite(data)
        absolute_mask = np.abs(data) <= 1e10
        valid_mask = valid_mask & finite_mask & absolute_mask
        
        if not np.any(valid_mask):
            return jsonify({'error': '没有有效的数据点'}), 400
        
        # 计算数据范围
        valid_data = data[valid_mask]
        data_min = np.min(valid_data)
        data_max = np.max(valid_data)
        print(f"数据范围: [{data_min:.6f}, {data_max:.6f}]")
        
        # 创建过滤后的数据数组
        data_filtered = np.where(valid_mask, data.astype(float), np.nan)
        
        # === 步骤1: 地球纹理预处理 ===
        print("\n=== 步骤1: 地球纹理预处理 ===")
        
        # 加载地球纹理
        earth_texture_path = os.path.join(os.path.dirname(__file__), 'cesium', 'Assets', 'Textures', 'earth_8k.jpg')
        if not os.path.exists(earth_texture_path):
            raise Exception(f"找不到地球纹理文件: {earth_texture_path}")
        
        # 使用全局缓存避免重复加载
        global _earth_texture_cache, _earth_texture_path_cache
        if _earth_texture_cache is None or _earth_texture_path_cache != earth_texture_path:
            print("加载地球纹理到缓存...")
            _earth_texture_cache = cv2.imread(earth_texture_path)
            _earth_texture_path_cache = earth_texture_path
            if _earth_texture_cache is None:
                raise Exception("无法读取地球纹理文件")
        else:
            print("使用缓存的地球纹理")
        
        earth_img = _earth_texture_cache
        earth_height, earth_width = earth_img.shape[:2]
        print(f"原始地球纹理尺寸: {earth_width}x{earth_height}")
        
        # 计算NC数据的经纬度分辨率
        lat_resolution = (lat_max - lat_min) / data_height
        lon_resolution = (lon_max_mapped - lon_min_mapped) / data_width
        print(f"NC数据分辨率: 纬度{lat_resolution:.6f}°/像素, 经度{lon_resolution:.6f}°/像素")
        
        # 计算全球在此分辨率下的尺寸
        if lon_resolution <= 0 or lat_resolution <= 0:
            return jsonify({'error': f'无效的分辨率: 经度{lon_resolution:.6f}°/像素, 纬度{lat_resolution:.6f}°/像素'}), 400
        
        global_width = data_width#int(360 / abs(lon_resolution))
        global_height = data_height#int(180 / abs(lat_resolution))
        
        # 验证计算出的尺寸
        if global_width <= 0 or global_height <= 0:
            return jsonify({'error': f'计算出的全球尺寸无效: {global_width}x{global_height}'}), 400
        
        print(f"全球在NC分辨率下的尺寸: {global_width}x{global_height}")
        
        # 将地球纹理调整到NC数据的分辨率
        earth_resized_global = cv2.resize(earth_img, (global_width, global_height), interpolation=cv2.INTER_LINEAR)
        print(f"地球纹理已调整到NC分辨率: {global_width}x{global_height}")
        
        # 计算NC数据区域在调整后地球纹理中的位置
        # 统一使用[-180, 180]格式的经度映射
        x_start = int((lon_min_mapped + 180) / 360 * global_width)
        x_end = x_start + data_width
        
        y_start = int((90 - lat_max) / 180 * global_height)
        y_end = y_start + data_height
        
        # 确保索引在有效范围内
        x_start = max(0, min(x_start, global_width - data_width))
        x_end = min(x_start + data_width, global_width)
        y_start = max(0, min(y_start, global_height - data_height))
        y_end = min(y_start + data_height, global_height)
        
        print(f"从调整后地球纹理中提取区域: x[{x_start}:{x_end}], y[{y_start}:{y_end}]")
        
        # 提取对应区域
        earth_resized = earth_resized_global[y_start:y_end, x_start:x_end]
        
        # 如果提取的区域尺寸不完全匹配，进行最终调整
        if earth_resized.shape[:2] != (data_height, data_width):
            earth_resized = cv2.resize(earth_resized, (data_width, data_height), interpolation=cv2.INTER_LINEAR)
            print(f"最终调整地球纹理尺寸为: {data_width}x{data_height}")
        
        print(f"地球纹理已调整为NC数据分辨率: {earth_resized.shape[:2]}")
        print("=== 地球纹理预处理完成 ===")
        
        # === 步骤2: 处理NC数据并生成彩色图像 ===
        print("\n=== 步骤2: 处理NC数据并生成彩色图像 ===")
        
        # 处理颜色方案映射
        color_scheme_mapping = {
            'grayscale': 'gray',
            'grey': 'gray',
            'greys': 'Greys'
        }
        mapped_scheme = color_scheme_mapping.get(color_scheme, color_scheme)
        
        # 标准化数据到0-1范围
        print(f"标准化数据范围: [{data_min:.6f}, {data_max:.6f}]")
        norm = Normalize(vmin=data_min, vmax=data_max)
        
        normalized_data = np.full_like(data_filtered, np.nan)
        valid_data_mask = ~np.isnan(data_filtered)
        if np.any(valid_data_mask):
            normalized_data[valid_data_mask] = norm(data_filtered[valid_data_mask])
            print(f"归一化了 {np.sum(valid_data_mask)} 个有效数据点")
        
        # 应用颜色映射
        if custom_colormap and color_scheme == 'custom':
            print("使用自定义颜色映射")
            colored_data = np.zeros((normalized_data.shape[0], normalized_data.shape[1], 4), dtype=np.float32)
            valid_mask = ~np.isnan(normalized_data)
            if np.any(valid_mask):
                valid_values = normalized_data[valid_mask]
                rgb_colors = apply_custom_colormap_vectorized(valid_values, custom_colormap)
                colored_data[valid_mask, 0] = rgb_colors[:, 0] / 255.0  # R
                colored_data[valid_mask, 1] = rgb_colors[:, 1] / 255.0  # G
                colored_data[valid_mask, 2] = rgb_colors[:, 2] / 255.0  # B
                colored_data[valid_mask, 3] = 1.0  # A
        else:
            try:
                cmap = cm.get_cmap(mapped_scheme)
            except ValueError:
                print(f"警告: 颜色方案 '{color_scheme}' 不支持，使用默认的 'viridis'")
                cmap = cm.get_cmap('viridis')
            colored_data = cmap(normalized_data)
        
        # 转换为OpenCV格式 (BGR, 0-255)
        bgr_data = colored_data[:, :, [2, 1, 0]]  # RGBA -> BGR
        bgr_data = (bgr_data * 255).astype(np.uint8)
        
        # 创建alpha通道，NaN值为完全透明
        alpha_channel = np.ones((data_filtered.shape[0], data_filtered.shape[1]), dtype=np.uint8) * 255
        nan_mask = np.isnan(data_filtered)
        alpha_channel[nan_mask] = 0
        
        print(f"NC数据彩色图像生成完成: {bgr_data.shape}")
        print("=== NC数据处理完成 ===")
        
        # === 步骤3: 图像合成生成最终结果 ===
        print("\n=== 步骤3: 图像合成生成最终结果 ===")
        
        # 确保两个图像尺寸一致
        if bgr_data.shape[:2] != earth_resized.shape[:2]:
            print(f"调整图像尺寸匹配: {bgr_data.shape[:2]} -> {earth_resized.shape[:2]}")
            bgr_data = cv2.resize(bgr_data, (earth_resized.shape[1], earth_resized.shape[0]), interpolation=cv2.INTER_LINEAR)
            alpha_channel = cv2.resize(alpha_channel, (earth_resized.shape[1], earth_resized.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # 图像混合
        alpha_norm = alpha_channel.astype(np.float32) / 255.0
        alpha_3ch = np.stack([alpha_norm] * 3, axis=2)
        print(f"图像混合，使用alpha通道: {alpha_3ch.shape}")
        # bgr_data alpha_3ch转化为图片并保存
        cv2.imwrite('bgr_data.png', bgr_data)
        cv2.imwrite('alpha_3ch.png', alpha_3ch)
        cv2.imwrite('earth_resized.png', earth_resized)
        final_img = (bgr_data.astype(np.float32) * alpha_3ch + 
                    earth_resized.astype(np.float32) * (1 - alpha_3ch)).astype(np.uint8)
        cv2.imwrite('final_img.png', final_img)
        print(f"图像合成完成，最终尺寸: {final_img.shape}")
        print("=== 所有步骤完成 ===")
        
        # 编码为PNG格式
        success, img_encoded = cv2.imencode('.png', final_img)
        if not success:
            raise Exception("无法编码图片为PNG格式")
        
        # 转换为base64编码
        import base64
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
        
        # 获取图片分辨率
        img_height, img_width = final_img.shape[:2]
        
        # 返回JSON格式，包含图片数据和分辨率信息
        return jsonify({
            'image_data': f'data:image/png;base64,{img_base64}',
            'width': int(img_width),
            'height': int(img_height),
            'data_width': int(data_width),
            'data_height': int(data_height),
            'longitude_range': [float(lon_min_mapped), float(lon_max_mapped)],
            'latitude_range': [float(lat_min), float(lat_max)]
        })
        
    except Exception as e:
        print(f"生成可视化图片时出错: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'生成图片时出错: {str(e)}'}), 500

@app.route('/api/visualization/<variable_name>/colorbar', methods=['GET'])
def get_colorbar_info(variable_name):
    """获取颜色条信息"""
    global current_nc_data
    
    if not current_nc_data:
        return jsonify({'error': '没有打开的NC文件'}), 400
    
    try:
        if variable_name not in current_nc_data.variables:
            return jsonify({'error': f'变量 {variable_name} 不存在'}), 400
        
        color_scheme = request.args.get('color_scheme', 'viridis')
        custom_colormap_str = request.args.get('custom_colormap')
        custom_colormap = None
        
        # 解析自定义颜色映射
        if custom_colormap_str:
            try:
                custom_colormap = json.loads(custom_colormap_str)
                print(f"使用自定义颜色映射，包含 {len(custom_colormap)} 个颜色点")
            except json.JSONDecodeError:
                print("自定义颜色映射解析失败，使用默认颜色方案")
                custom_colormap = None
        
        # 获取数据范围（与图片生成保持一致）
        var_data = current_nc_data.variables[variable_name]
        
        # 获取维度参数
        time_index = request.args.get('time_index', type=int)
        depth_index = request.args.get('depth_index', type=int)
        level_index = request.args.get('level', type=int)
        # 支持im_level参数（与level参数等效）
        if level_index is None:
            level_index = request.args.get('im_level', type=int)
        
        # 根据维度提取数据（与图片生成函数保持一致）
        if len(var_data.shape) == 4:  # (time, depth, lat, lon)
            t_idx = time_index if time_index is not None else 0
            # 优先使用level_index，其次是depth_index
            d_idx = level_index if level_index is not None else (depth_index if depth_index is not None else 0)
            data = var_data[t_idx, d_idx, :, :]
            print(f"提取4D数据: time_index={t_idx}, level/depth_index={d_idx}")
        elif len(var_data.shape) == 3:
            if time_index is not None:
                data = var_data[time_index, :, :]
                print(f"提取3D数据: time_index={time_index}")
            elif level_index is not None:
                data = var_data[level_index, :, :]
                print(f"提取3D数据: level_index={level_index}")
            elif depth_index is not None:
                data = var_data[depth_index, :, :]
                print(f"提取3D数据: depth_index={depth_index}")
            else:
                data = var_data[0, :, :]
                print("提取3D数据: 使用默认索引0")
        elif len(var_data.shape) == 2:  # (lat, lon)
            data = var_data[:, :]
        else:
            return jsonify({'error': f'不支持的数据维度: {var_data.shape}'}), 400
        
        # 转换为numpy数组并处理异常值（与图片生成保持一致）
        data = np.array(data)
        
        # 沿着纬度方向翻转数据（与图片生成保持一致）
        data = np.flipud(data)
        
        # 处理 masked array 和 NaN 值
        if hasattr(data, 'mask'):
            valid_mask = ~data.mask
            data = data.data
        else:
            valid_mask = np.ones(data.shape, dtype=bool)
        
        # 过滤异常值（与图片生成保持一致，只做基础过滤）
        finite_mask = np.isfinite(data)
        valid_mask = valid_mask & finite_mask
        
        # 应用绝对值上限过滤（10^10）
        absolute_limit = 1e10
        absolute_mask = np.abs(data) <= absolute_limit
        valid_mask = valid_mask & absolute_mask
        
        if not np.any(valid_mask):
            return jsonify({'error': '没有有效的数据点'}), 400
        
        # 计算数据范围（与图片生成保持一致）
        valid_data = data[valid_mask]
        data_min = np.min(valid_data)
        data_max = np.max(valid_data)
        
        print(f"Colorbar数据范围: [{data_min:.6f}, {data_max:.6f}]")
        
        # 处理颜色方案映射
        color_scheme_mapping = {
            'grayscale': 'gray',  # 将grayscale映射到matplotlib的gray
            'grey': 'gray',       # 统一使用gray
            'greys': 'Greys'      # 大写版本
        }
        
        # 如果是自定义的颜色方案名称，映射到matplotlib支持的名称
        mapped_scheme = color_scheme_mapping.get(color_scheme, color_scheme)
        
        # 生成颜色条
        colorbar_steps = []
        num_steps = 20
        
        if custom_colormap and color_scheme == 'custom':
            # 使用自定义颜色映射
            for i in range(num_steps):
                normalized_value = i / (num_steps - 1)
                actual_value = data_min + normalized_value * (data_max - data_min)
                color = get_color_from_custom_colormap(normalized_value, custom_colormap)
                
                colorbar_steps.append({
                    'value': actual_value,
                    'normalized': normalized_value,
                    'color': color
                })
        else:
            # 使用matplotlib颜色映射
            try:
                cmap = cm.get_cmap(mapped_scheme)
            except ValueError:
                print(f"警告: 颜色方案 '{color_scheme}' 不支持，使用默认的 'viridis'")
                cmap = cm.get_cmap('viridis')
            
            for i in range(num_steps):
                normalized_value = i / (num_steps - 1)
                actual_value = data_min + normalized_value * (data_max - data_min)
                color = cmap(normalized_value)
                
                colorbar_steps.append({
                    'value': actual_value,
                    'normalized': normalized_value,
                    'color': [int(c * 255) for c in color[:3]]  # RGB
                })
        
        return jsonify({
            'success': True,
            'data_range': {
                'min': float(data_min),
                'max': float(data_max)
            },
            'colorbar': colorbar_steps,
            'color_scheme': color_scheme
        })
        
    except Exception as e:
        print(f"获取颜色条信息时出错: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'获取颜色条信息时出错: {str(e)}'}), 500

@app.route('/api/visualization/<variable_name>/info', methods=['GET'])
def get_visualization_info(variable_name):
    """获取可视化数据的概览信息，不返回具体数据点"""
    global current_nc_data
    
    if not current_nc_data:
        return jsonify({'error': '没有打开的NC文件'}), 400
    
    try:
        if variable_name not in current_nc_data.variables:
            return jsonify({'error': f'变量 {variable_name} 不存在'}), 400
        
        # 获取参数
        time_index = request.args.get('time_index', type=int)
        depth_index = request.args.get('depth_index', type=int)
        level_index = request.args.get('level', type=int)
        # 支持im_level参数（与level参数等效）
        if level_index is None:
            level_index = request.args.get('im_level', type=int)
        
        # 获取变量数据
        var_data = current_nc_data.variables[variable_name]
        
        # 获取坐标信息
        coordinates = extract_coordinates(current_nc_data)
        if not coordinates:
            return jsonify({'error': '无法提取坐标信息'}), 400
        
        latitudes = np.array(coordinates['latitudes'])
        longitudes = np.array(coordinates['longitudes'])
        
        # 根据维度提取数据（支持level参数）
        if len(var_data.shape) == 4:  # (time, depth/level, lat, lon)
            t_idx = time_index if time_index is not None else 0
            # 优先使用level_index，其次是depth_index
            d_idx = level_index if level_index is not None else (depth_index if depth_index is not None else 0)
            data = var_data[t_idx, d_idx, :, :]
            print(f"提取4D数据: time_index={t_idx}, level/depth_index={d_idx}")
        elif len(var_data.shape) == 3:
            if time_index is not None:
                data = var_data[time_index, :, :]
                print(f"提取3D数据: time_index={time_index}")
            elif level_index is not None:
                data = var_data[level_index, :, :]
                print(f"提取3D数据: level_index={level_index}")
            elif depth_index is not None:
                data = var_data[depth_index, :, :]
                print(f"提取3D数据: depth_index={depth_index}")
            else:
                data = var_data[0, :, :]
                print("提取3D数据: 使用默认索引0")
        elif len(var_data.shape) == 2:  # (lat, lon)
            data = var_data[:, :]
            print("提取2D数据")
        else:
            return jsonify({'error': f'不支持的数据维度: {var_data.shape}'}), 400
        
        # 转换为numpy数组并处理masked array
        data = np.array(data)
        
        # 处理 masked array 和 NaN 值
        if hasattr(data, 'mask'):
            valid_mask = ~data.mask
            data = data.data
        else:
            valid_mask = np.ones(data.shape, dtype=bool)
        
        # 进一步过滤 NaN、无穷大和异常值
        finite_mask = np.isfinite(data)
        valid_mask = valid_mask & finite_mask
        
        # 使用四分位数方法过滤极端异常值
        if np.any(valid_mask):
            valid_data = data[valid_mask]
            q1 = np.percentile(valid_data, 25)
            q3 = np.percentile(valid_data, 75)
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            
            outlier_mask = (data >= lower_bound) & (data <= upper_bound)
            valid_mask = valid_mask & outlier_mask
        
        # 计算有效数据的统计信息
        if np.any(valid_mask):
            valid_data = data[valid_mask]
            data_min = float(np.min(valid_data))
            data_max = float(np.max(valid_data))
            total_valid_count = np.sum(valid_mask)
        else:
            return jsonify({'error': '没有有效的数据点'}), 400
        
        # 返回概览信息
        result = {
            'success': True,
            'variable_name': variable_name,
            'total_points': int(total_valid_count),
            'data_range': {
                'min': data_min,
                'max': data_max
            },
            'spatial_extent': {
                'lat_min': float(np.min(latitudes)),
                'lat_max': float(np.max(latitudes)),
                'lon_min': float(np.min(longitudes)),
                'lon_max': float(np.max(longitudes))
            },
            'grid_size': {
                'lat_count': len(latitudes),
                'lon_count': len(longitudes)
            },
            'metadata': {
                'units': getattr(var_data, 'units', ''),
                'long_name': getattr(var_data, 'long_name', variable_name),
                'original_shape': list(var_data.shape)
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"获取可视化概览信息错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'获取概览信息失败: {str(e)}'}), 500


@app.route('/api/data/<variable_name>', methods=['GET'])
def get_variable_data_json(variable_name):
    """获取变量数据的JSON格式，供前端直接渲染"""
    global current_nc_data
    
    if not current_nc_data:
        return jsonify({'error': '没有打开的NC文件'}), 400
    
    try:
        if variable_name not in current_nc_data.variables:
            return jsonify({'error': f'变量 {variable_name} 不存在'}), 400
        
        # 获取变量数据
        var_data = current_nc_data.variables[variable_name]
        
        # 获取坐标信息
        coordinates = extract_coordinates(current_nc_data)
        if not coordinates:
            return jsonify({'error': '无法提取坐标信息'}), 400
        
        # 获取数据（取第一个时间步和深度层）
        if len(var_data.shape) == 4:  # (time, depth, lat, lon)
            data = var_data[0, 0, :, :]
        elif len(var_data.shape) == 3:  # (time, lat, lon)
            data = var_data[0, :, :]
        elif len(var_data.shape) == 2:  # (lat, lon)
            data = var_data[:, :]
        else:
            return jsonify({'error': f'不支持的数据维度: {var_data.shape}'}), 400
        
        # 转换为numpy数组
        data = np.array(data)
        
        # 处理masked array
        if hasattr(data, 'mask'):
            # 保存原始掩码信息
            original_mask = data.mask
            data = np.ma.filled(data, np.nan)
        else:
            original_mask = None
        
        # 检测填充值和无效值
        fill_value_threshold = 1e30
        invalid_mask = np.isnan(data) | np.isinf(data) | (np.abs(data) > fill_value_threshold)
        
        # 如果有原始掩码，合并掩码信息
        if original_mask is not None:
            invalid_mask = invalid_mask | original_mask
        
        # 创建有效数据掩码（不排除0值，只排除真正的无效值）
        valid_mask = ~invalid_mask
        
        if not np.any(valid_mask):
            # 如果所有数据都是无效值，返回错误
            return jsonify({'error': '数据中没有有效值'}), 400
        else:
            # 计算有效数据的范围
            valid_data = data[valid_mask]
            data_min = float(np.min(valid_data))
            data_max = float(np.max(valid_data))
            
            # 将无效值设置为NaN（前端会处理）
            data[invalid_mask] = np.nan
        
        # 准备返回的数据
        # 处理NaN值以确保JSON序列化正确
        data_list = data.tolist()
        # 将NaN值替换为None（在JSON中会变成null）
        def replace_nan(obj):
            if isinstance(obj, list):
                return [replace_nan(item) for item in obj]
            elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return None
            else:
                return obj
        
        data_list = replace_nan(data_list)
        
        result = {
            'success': True,
            'variable_name': variable_name,
            'coordinates': {
                'longitudes': coordinates['longitudes'],
                'latitudes': coordinates['latitudes']
            },
            'data': {
                'values': data_list,
                'min': data_min,
                'max': data_max,
                'shape': data.shape
            },
            'valid_mask': valid_mask.tolist(),
            'units': getattr(var_data, 'units', ''),
            'long_name': getattr(var_data, 'long_name', variable_name)
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"获取变量数据错误: {e}")
        print(traceback.format_exc())
        return jsonify({'error': f'获取数据失败: {str(e)}'}), 500

@app.route('/api/close', methods=['POST'])
def close_file():
    """关闭当前文件"""
    global current_nc_data, current_file_path
    
    try:
        if current_nc_data:
            current_nc_data.close()
            current_nc_data = None
        
        if current_file_path and os.path.exists(current_file_path):
            os.remove(current_file_path)
            current_file_path = None
        
        return jsonify({'success': True, 'message': '文件已关闭'})
        
    except Exception as e:
        print(f"关闭文件错误: {e}")
        return jsonify({'error': f'关闭文件失败: {str(e)}'}), 500

if __name__ == '__main__':
    print("🌍 3D NC文件可视化器 - 后端服务启动")
    print("=" * 50)
    print(f"📍 服务器地址: http://localhost:8080")
    print(f"📁 上传目录: {UPLOAD_FOLDER}")
    print(f"💡 支持的文件格式: .nc")
    print(f"📊 最大文件大小: 5GB")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=8080)