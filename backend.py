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
from PIL import Image
import io
import base64
import time

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
                attributes[attr_name] = attr_value
            except:
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
            info['global_attributes'][attr_name] = attr_value
        except:
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
            """应用颜色映射到归一化数据 - 向量化优化版本"""
            import concurrent.futures
            from functools import partial
            
            colormaps = {
                'viridis': np.array([
                    [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
                    [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37]
                ], dtype=np.float32),
                'plasma': np.array([
                    [13, 8, 135], [75, 3, 161], [125, 3, 168], [168, 34, 150], [203, 70, 121],
                    [229, 107, 93], [248, 148, 65], [253, 195, 40], [239, 248, 33]
                ], dtype=np.float32),
                'coolwarm': np.array([
                    [59, 76, 192], [98, 130, 234], [141, 176, 254], [184, 208, 249], [221, 221, 221],
                    [245, 183, 142], [235, 127, 96], [215, 48, 39], [165, 0, 38]
                ], dtype=np.float32),
                'jet': np.array([
                    [0, 0, 143], [0, 0, 255], [0, 127, 255], [0, 255, 255], [127, 255, 127],
                    [255, 255, 0], [255, 127, 0], [255, 0, 0], [127, 0, 0]
                ], dtype=np.float32)
            }
            
            colors = colormaps.get(colormap_name, colormaps['viridis'])
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
            
            # 设置无效数据为0（黑色）
            gray_data[~valid_mask] = 0
            
            # 翻转Y轴（图像坐标系与地理坐标系相反）
            gray_data = np.flipud(gray_data)
            
            # 创建PIL灰度图像
            img = Image.fromarray(gray_data, mode='L')
            
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