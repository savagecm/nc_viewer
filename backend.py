#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D NC文件可视化器 - Flask后端服务
处理NC文件上传、解析和数据提供
"""

from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from flask_cors import CORS
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
        
        # 应用颜色映射（优化版本，使用向量化操作）
        def apply_colormap(data, colormap_name):
            """应用颜色映射到归一化数据 - 使用matplotlib内置颜色映射"""
            import matplotlib.cm as cm
            
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
            print(f"开始生成彩色图，数据尺寸: {normalized_data.shape}，颜色方案: {color_scheme}")
            start_time = time.time()
            
            rgb_data = apply_colormap(normalized_data, color_scheme)
            
            # 翻转Y轴（图像坐标系与地理坐标系相反）
            rgb_data = np.flipud(rgb_data)
            
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
        opacity = float(request.args.get('opacity', 0.8))
        
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
        
        print(f"开始处理变量 {variable_name} 的可视化数据")
        print(f"颜色方案: {color_scheme}, 透明度: {opacity}")
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
            rgb_color = get_color_for_value(normalized_value, color_scheme)
            
            visualization_points.append({
                'longitude': point['lon'],
                'latitude': point['lat'],
                'value': point['value'],
                'color': rgb_color,
                'opacity': opacity
            })
        
        # 生成颜色条数据
        colorbar_data = []
        num_steps = 50
        for i in range(num_steps):
            normalized_value = i / (num_steps - 1)
            value = data_min + normalized_value * (data_max - data_min)
            rgb_color = get_color_for_value(normalized_value, color_scheme)
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
            'opacity': opacity,
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

def get_color_for_value(normalized_value, scheme='viridis'):
    """根据归一化值和颜色方案获取RGB颜色 - 使用matplotlib内置颜色映射"""
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


@app.route('/api/visualization/<variable_name>/image', methods=['GET'])
def generate_visualization_image(variable_name):
    """生成可视化图片，包含数据、透明度、底图和colorbar"""
    global current_nc_data
    
    if not current_nc_data:
        return jsonify({'error': '没有打开的NC文件'}), 400
    
    try:
        if variable_name not in current_nc_data.variables:
            return jsonify({'error': f'变量 {variable_name} 不存在'}), 400
        
        # 获取参数
        color_scheme = request.args.get('color_scheme', 'viridis')
        opacity = float(request.args.get('opacity', 0.8))
        
        # 注意：图片尺寸将完全基于NC数据的网格大小，忽略URL中的width和height参数
        
        # 获取维度参数
        time_index = request.args.get('time_index', type=int)
        depth_index = request.args.get('depth_index', type=int)
        
        print(f"生成变量 {variable_name} 的可视化图片")
        print(f"参数: 颜色方案={color_scheme}, 透明度={opacity}")
        
        # 获取变量数据
        var_data = current_nc_data.variables[variable_name]
        
        # 获取坐标信息
        coordinates = extract_coordinates(current_nc_data)
        if not coordinates:
            return jsonify({'error': '无法提取坐标信息'}), 400
        
        latitudes = np.array(coordinates['latitudes'])
        longitudes = np.array(coordinates['longitudes'])
        
        # 根据维度提取数据
        if len(var_data.shape) == 4:  # (time, depth, lat, lon)
            t_idx = time_index if time_index is not None else 0
            d_idx = depth_index if depth_index is not None else 0
            data = var_data[t_idx, d_idx, :, :]
        elif len(var_data.shape) == 3:
            if time_index is not None:
                data = var_data[time_index, :, :]
            elif depth_index is not None:
                data = var_data[depth_index, :, :]
            else:
                data = var_data[0, :, :]
        elif len(var_data.shape) == 2:  # (lat, lon)
            data = var_data[:, :]
        else:
            return jsonify({'error': f'不支持的数据维度: {var_data.shape}'}), 400
        
        # 根据数据的实际经纬度维度设置图片尺寸
        # 每个数据点对应一个像素点，不使用用户传入的尺寸参数
        data_height, data_width = data.shape  # lat, lon
        width = data_width   # 经度方向
        height = data_height # 纬度方向
            
        print(f"数据维度: {data_height}x{data_width} (纬度x经度)")
        print(f"输出图片尺寸: {width}x{height} (1:1像素映射)")
        
        # 转换为numpy数组并处理异常值
        data = np.array(data)
        
        # 沿着纬度方向翻转数据（通常NC文件的纬度是从北到南，需要翻转为从南到北）
        data = np.flipud(data)
        print(f"数据已沿纬度方向翻转")
        
        # 处理 masked array 和 NaN 值
        if hasattr(data, 'mask'):
            valid_mask = ~data.mask
            data = data.data
        else:
            valid_mask = np.ones(data.shape, dtype=bool)
        
        # 过滤异常值
        finite_mask = np.isfinite(data)
        valid_mask = valid_mask & finite_mask
        
        # 应用绝对值上限过滤（10^10）
        absolute_limit = 1e10
        absolute_mask = np.abs(data) <= absolute_limit
        valid_mask = valid_mask & absolute_mask
        
        if np.any(valid_mask):
            valid_data = data[valid_mask]
            
            # 使用百分位数方法移除极端值
            p1 = np.percentile(valid_data, 1)
            p99 = np.percentile(valid_data, 99)
            
            # 过滤掉超出1%-99%范围的极端值
            percentile_mask = (data >= p1) & (data <= p99)
            valid_mask = valid_mask & percentile_mask
            
            if np.any(valid_mask):
                filtered_data = data[valid_mask]
                q1 = np.percentile(filtered_data, 25)
                q3 = np.percentile(filtered_data, 75)
                iqr = q3 - q1
                
                # 使用1.5倍IQR标准
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                final_outlier_mask = (data >= lower_bound) & (data <= upper_bound)
                valid_mask = valid_mask & final_outlier_mask
        
        if not np.any(valid_mask):
            return jsonify({'error': '没有有效的数据点'}), 400
        
        # 将无效数据设为NaN
        data_filtered = data.copy().astype(float)
        data_filtered[~valid_mask] = np.nan
        
        # 计算数据范围
        valid_data = data_filtered[valid_mask]
        data_min = np.min(valid_data)
        data_max = np.max(valid_data)
        
        print(f"数据范围: [{data_min:.6f}, {data_max:.6f}]")
        
        # 获取数据的实际经纬度范围
        coordinates = extract_coordinates(current_nc_data)
        if not coordinates:
            raise Exception('无法提取坐标信息')
        
        latitudes = np.array(coordinates['latitudes'])
        longitudes = np.array(coordinates['longitudes'])
        
        # 如果数据被翻转了，纬度也需要相应调整
        latitudes = np.flipud(latitudes)
        
        lat_min, lat_max = float(np.min(latitudes)), float(np.max(latitudes))
        lon_min, lon_max = float(np.min(longitudes)), float(np.max(longitudes))
        
        print(f"数据地理范围: 纬度[{lat_min:.2f}, {lat_max:.2f}], 经度[{lon_min:.2f}, {lon_max:.2f}]")
        
        # 加载地球底图
        earth_texture_path = os.path.join(os.path.dirname(__file__), 'cesium', 'Assets', 'Textures', 'earth_8k.jpg')
        if not os.path.exists(earth_texture_path):
            raise Exception(f"找不到地球纹理文件: {earth_texture_path}")
        
        # 读取地球纹理
        earth_img = cv2.imread(earth_texture_path)
        if earth_img is None:
            raise Exception("无法读取地球纹理文件")
        
        # 地球纹理的尺寸和地理范围
        earth_height, earth_width = earth_img.shape[:2]
        
        # 计算数据区域在地球纹理中的像素位置
        # 地球纹理覆盖全球：经度[-180, 180]，纬度[-90, 90]
        # 注意：图片的Y轴是从上到下，对应纬度从90到-90
        
        # 经度映射：[-180, 180] -> [0, earth_width]
        x_start = int((lon_min + 180) / 360 * earth_width)
        x_end = int((lon_max + 180) / 360 * earth_width)
        
        # 纬度映射：[90, -90] -> [0, earth_height]
        y_start = int((90 - lat_max) / 180 * earth_height)
        y_end = int((90 - lat_min) / 180 * earth_height)
        
        # 确保索引在有效范围内
        x_start = max(0, min(x_start, earth_width - 1))
        x_end = max(x_start + 1, min(x_end, earth_width))
        y_start = max(0, min(y_start, earth_height - 1))
        y_end = max(y_start + 1, min(y_end, earth_height))
        
        print(f"地球纹理裁剪区域: x[{x_start}:{x_end}], y[{y_start}:{y_end}]")
        
        # 从地球纹理中提取对应区域
        earth_region = earth_img[y_start:y_end, x_start:x_end]
        
        # 调整提取的区域大小到数据尺寸
        earth_resized = cv2.resize(earth_region, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # 处理颜色方案映射
        color_scheme_mapping = {
            'grayscale': 'gray',  # 将grayscale映射到matplotlib的gray
            'grey': 'gray',       # 统一使用gray
            'greys': 'Greys'      # 大写版本
        }
        
        # 如果是自定义的颜色方案名称，映射到matplotlib支持的名称
        mapped_scheme = color_scheme_mapping.get(color_scheme, color_scheme)
        
        # 获取colormap
        try:
            cmap = cm.get_cmap(mapped_scheme)
        except ValueError:
            print(f"警告: 颜色方案 '{color_scheme}' 不支持，使用默认的 'viridis'")
            cmap = cm.get_cmap('viridis')
        
        # 标准化数据到0-1范围
        norm = Normalize(vmin=data_min, vmax=data_max)
        normalized_data = norm(data_filtered)
        
        # 应用colormap
        colored_data = cmap(normalized_data)
        
        # 转换为OpenCV格式 (BGR, 0-255)
        # matplotlib colormap输出RGBA，需要转换为BGR
        bgr_data = colored_data[:, :, [2, 1, 0]]  # RGBA -> BGR
        bgr_data = (bgr_data * 255).astype(np.uint8)
        
        # 创建alpha通道，NaN值为完全透明
        alpha_channel = np.ones((data_filtered.shape[0], data_filtered.shape[1]), dtype=np.uint8) * 255
        nan_mask = np.isnan(data_filtered)
        alpha_channel[nan_mask] = 0  # NaN值设为完全透明
        
        # 应用用户设置的透明度
        alpha_channel = (alpha_channel * opacity).astype(np.uint8)
        
        # 调整数据图片大小以匹配目标尺寸
        bgr_resized = cv2.resize(bgr_data, (width, height), interpolation=cv2.INTER_LINEAR)
        alpha_resized = cv2.resize(alpha_channel, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # 将NC数据叠加到地球纹理上
        final_img = earth_resized.copy().astype(np.float32)
        
        # 按像素混合
        alpha_norm = alpha_resized.astype(np.float32) / 255.0
        for c in range(3):  # BGR三个通道
            final_img[:, :, c] = (bgr_resized[:, :, c].astype(np.float32) * alpha_norm + 
                                 final_img[:, :, c] * (1 - alpha_norm))
        
        final_img = final_img.astype(np.uint8)
        
        # 编码为PNG格式
        success, img_encoded = cv2.imencode('.png', final_img)
        if not success:
            raise Exception("无法编码图片为PNG格式")
        
        # 创建BytesIO对象
        img_buffer = io.BytesIO(img_encoded.tobytes())
        
        img_buffer.seek(0)
        
        # 返回图片
        return send_file(img_buffer, mimetype='image/png')
        
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
        
        # 获取数据范围（简化版本，不处理所有数据）
        var_data = current_nc_data.variables[variable_name]
        
        # 获取维度参数
        time_index = request.args.get('time_index', type=int)
        depth_index = request.args.get('depth_index', type=int)
        
        # 根据维度提取数据样本
        if len(var_data.shape) == 4:  # (time, depth, lat, lon)
            t_idx = time_index if time_index is not None else 0
            d_idx = depth_index if depth_index is not None else 0
            sample_data = var_data[t_idx, d_idx, ::10, ::10]  # 采样
        elif len(var_data.shape) == 3:
            if time_index is not None:
                sample_data = var_data[time_index, ::10, ::10]
            elif depth_index is not None:
                sample_data = var_data[depth_index, ::10, ::10]
            else:
                sample_data = var_data[0, ::10, ::10]
        elif len(var_data.shape) == 2:  # (lat, lon)
            sample_data = var_data[::10, ::10]
        else:
            return jsonify({'error': f'不支持的数据维度: {var_data.shape}'}), 400
        
        # 转换为numpy数组并处理异常值
        sample_data = np.array(sample_data)
        
        # 处理 masked array 和 NaN 值
        if hasattr(sample_data, 'mask'):
            valid_mask = ~sample_data.mask
            sample_data = sample_data.data
        else:
            valid_mask = np.ones(sample_data.shape, dtype=bool)
        
        # 过滤异常值
        finite_mask = np.isfinite(sample_data)
        valid_mask = valid_mask & finite_mask
        
        # 应用绝对值上限过滤（10^10）
        absolute_limit = 1e10
        absolute_mask = np.abs(sample_data) <= absolute_limit
        valid_mask = valid_mask & absolute_mask
        
        if np.any(valid_mask):
            valid_data = sample_data[valid_mask]
            
            # 使用百分位数方法移除极端值
            p1 = np.percentile(valid_data, 1)
            p99 = np.percentile(valid_data, 99)
            
            # 过滤掉超出1%-99%范围的极端值
            percentile_mask = (sample_data >= p1) & (sample_data <= p99)
            valid_mask = valid_mask & percentile_mask
            
            if np.any(valid_mask):
                filtered_data = sample_data[valid_mask]
                q1 = np.percentile(filtered_data, 25)
                q3 = np.percentile(filtered_data, 75)
                iqr = q3 - q1
                
                # 使用1.5倍IQR标准
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                final_outlier_mask = (sample_data >= lower_bound) & (sample_data <= upper_bound)
                valid_mask = valid_mask & final_outlier_mask
        
        if not np.any(valid_mask):
            return jsonify({'error': '没有有效的数据点'}), 400
        
        # 计算数据范围
        valid_data = sample_data[valid_mask]
        data_min = np.min(valid_data)
        data_max = np.max(valid_data)
        
        # 处理颜色方案映射
        color_scheme_mapping = {
            'grayscale': 'gray',  # 将grayscale映射到matplotlib的gray
            'grey': 'gray',       # 统一使用gray
            'greys': 'Greys'      # 大写版本
        }
        
        # 如果是自定义的颜色方案名称，映射到matplotlib支持的名称
        mapped_scheme = color_scheme_mapping.get(color_scheme, color_scheme)
        
        # 生成颜色条
        try:
            cmap = cm.get_cmap(mapped_scheme)
        except ValueError:
            print(f"警告: 颜色方案 '{color_scheme}' 不支持，使用默认的 'viridis'")
            cmap = cm.get_cmap('viridis')
        colorbar_steps = []
        num_steps = 20
        
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
        
        # 获取变量数据
        var_data = current_nc_data.variables[variable_name]
        
        # 获取坐标信息
        coordinates = extract_coordinates(current_nc_data)
        if not coordinates:
            return jsonify({'error': '无法提取坐标信息'}), 400
        
        latitudes = np.array(coordinates['latitudes'])
        longitudes = np.array(coordinates['longitudes'])
        
        # 根据维度提取数据
        if len(var_data.shape) == 4:  # (time, depth, lat, lon)
            t_idx = time_index if time_index is not None else 0
            d_idx = depth_index if depth_index is not None else 0
            data = var_data[t_idx, d_idx, :, :]
        elif len(var_data.shape) == 3:
            if time_index is not None:
                data = var_data[time_index, :, :]
            elif depth_index is not None:
                data = var_data[depth_index, :, :]
            else:
                data = var_data[0, :, :]
        elif len(var_data.shape) == 2:  # (lat, lon)
            data = var_data[:, :]
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