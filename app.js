// 全局变量
let viewer;
let currentDataSource;
let currentVariable;
let fileInfo;
let colorbarCanvas;
let colorbarCtx;

// API基础URL
const API_BASE = '/api';

// 初始化Cesium viewer
function initializeCesium() {
    console.log('开始初始化Cesium...');
    
    try {
        // 设置Cesium基础路径（离线模式）
        window.CESIUM_BASE_URL = './cesium/';
        
        // 检查Cesium是否加载
        if (typeof Cesium === 'undefined') {
            throw new Error('Cesium库未加载');
        }
        
        // 禁用Cesium Ion默认访问令牌（使用空字符串而不是undefined）
        Cesium.Ion.defaultAccessToken = '';
        
        // 创建viewer
        viewer = new Cesium.Viewer('cesiumContainer', {
            homeButton: true,
            sceneModePicker: true,
            baseLayerPicker: false, // 禁用在线图层选择器
            navigationHelpButton: true,
            animation: false,
            timeline: false,
            fullscreenButton: true,
            vrButton: false,
            // 使用默认地形（离线模式）
            terrainProvider: new Cesium.EllipsoidTerrainProvider()
            // 不指定imageryProvider，让Cesium使用默认影像
        });
        
        // 移除默认影像层并添加我们的地球纹理
        viewer.imageryLayers.removeAll();
        
        // 尝试加载earth_8k.jpg作为地球纹理
        console.log('正在尝试加载地球纹理...');
        
        const earthImageryProvider = new Cesium.SingleTileImageryProvider({
            url: './cesium/Assets/Textures/earth_8k.jpg',
            rectangle: Cesium.Rectangle.fromDegrees(-180.0, -90.0, 180.0, 90.0)
        });
        
        // 添加错误处理
        if (earthImageryProvider.errorEvent) {
            earthImageryProvider.errorEvent.addEventListener(function(error) {
                console.error('地球纹理加载失败:', error);
            });
        }
        
        // 安全检查readyPromise是否存在
        if (earthImageryProvider.readyPromise) {
            earthImageryProvider.readyPromise.then(function() {
                console.log('地球纹理加载成功!');
            }).catch(function(error) {
                console.error('地球纹理加载Promise失败:', error);
            });
        } else {
            console.log('地球纹理提供者没有readyPromise属性');
        }
        
        viewer.imageryLayers.addImageryProvider(earthImageryProvider);
        
        // 设置相机初始位置
        viewer.camera.setView({
            destination: Cesium.Cartesian3.fromDegrees(0, 0, 20000000)
        });
        
        // 启用深度测试
        viewer.scene.globe.depthTestAgainstTerrain = false;
        
        // 确保底图显示
        const baseLayer = viewer.imageryLayers.get(0);
        if (baseLayer) {
            baseLayer.show = true;
            console.log('基础地图层已设置为显示');
        } else {
            console.warn('未找到基础地图层');
        }
        
        // 添加地图层加载事件监听
        viewer.imageryLayers.layerAdded.addEventListener(function(layer) {
            console.log('地图层已添加:', layer);
        });
        
        viewer.imageryLayers.layerRemoved.addEventListener(function(layer) {
            console.log('地图层已移除:', layer);
        });
        
        console.log('Cesium初始化成功，地图层数量:', viewer.imageryLayers.length);
        console.log('基础地图提供者:', viewer.imageryLayers.get(0)?.imageryProvider?.constructor?.name);
        
    } catch (error) {
        console.error('Cesium初始化失败:', error);
        alert('3D地球初始化失败: ' + error.message + '\n\n请确保网络连接正常或使用本地Cesium库');
    }
}

// 上传NC文件
function uploadNCFile(file) {
    console.log('开始上传NC文件:', file.name);
    
    const formData = new FormData();
    formData.append('file', file);
    
    // 显示加载状态
    const loadButton = document.getElementById('loadFile');
    const originalText = loadButton.textContent;
    loadButton.textContent = '上传中...';
    loadButton.disabled = true;
    
    fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            console.log('文件上传成功:', data);
            fileInfo = data.file_info;
            displayFileInfo(data.file_info);
            loadVariables();
            document.getElementById('variableSection').style.display = 'block';
        } else {
            throw new Error(data.error || '上传失败');
        }
    })
    .catch(error => {
        console.error('上传错误:', error);
        alert('文件上传失败: ' + error.message);
    })
    .finally(() => {
        loadButton.textContent = originalText;
        loadButton.disabled = false;
    });
}

// 显示文件信息
function displayFileInfo(info) {
    const infoDiv = document.getElementById('fileInfo');
    let html = `<strong>文件信息:</strong><br>`;
    
    // 显示维度信息
    html += `<br><strong>维度:</strong><br>`;
    for (const [dimName, dimInfo] of Object.entries(info.dimensions)) {
        html += `${dimName}: ${dimInfo.size}<br>`;
    }
    
    // 显示变量信息
    html += `<br><strong>变量 (${info.variables.length}):</strong><br>`;
    info.variables.forEach(variable => {
        html += `${variable.name} [${variable.dimensions.join(', ')}]<br>`;
    });
    
    // 显示全局属性
    if (Object.keys(info.global_attributes).length > 0) {
        html += `<br><strong>全局属性:</strong><br>`;
        for (const [attrName, attrValue] of Object.entries(info.global_attributes)) {
            html += `${attrName}: ${attrValue}<br>`;
        }
    }
    
    infoDiv.innerHTML = html;
}

// 加载变量列表
function loadVariables() {
    fetch(`${API_BASE}/variables`)
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.variables) {
            populateVariableSelect(data.variables);
        } else {
            throw new Error(data.error || '获取变量失败');
        }
    })
    .catch(error => {
        console.error('加载变量错误:', error);
        alert('加载变量列表失败: ' + error.message);
    });
}

// 填充变量选择下拉框
function populateVariableSelect(variables) {
    const select = document.getElementById('variableSelect');
    select.innerHTML = '<option value="">选择变量</option>';
    
    // 存储变量信息以便后续使用
    window.variablesData = {};
    
    variables.forEach(variable => {
        // 存储变量信息
        window.variablesData[variable.name] = variable;
        
        const option = document.createElement('option');
        option.value = variable.name;
        option.textContent = `${variable.name} [${variable.dimensions.join(', ')}]`;
        if (variable.long_name && variable.long_name !== variable.name) {
            option.textContent += ` - ${variable.long_name}`;
        }
        select.appendChild(option);
    });
    
    // 添加变量选择事件监听器
    select.addEventListener('change', function() {
        const selectedVariable = this.value;
        if (selectedVariable && window.variablesData[selectedVariable]) {
            showDimensionControls(window.variablesData[selectedVariable]);
        } else {
            hideDimensionControls();
        }
    });
}

// 显示维度控制
function showDimensionControls(variable) {
    const dimensionSection = document.getElementById('dimensionSection');
    const dimensionControls = document.getElementById('dimensionControls');
    
    // 清空现有控件
    dimensionControls.innerHTML = '';
    
    // 只为超过2维的变量显示维度控制
    if (variable.dimension_info.length > 2) {
        variable.dimension_info.forEach((dim, index) => {
            // 跳过最后两个维度（通常是lat/lon）
            if (index < variable.dimension_info.length - 2) {
                const dimControl = document.createElement('div');
                dimControl.style.marginBottom = '5px';
                
                const label = document.createElement('label');
                label.textContent = `${dim.name} (0-${dim.size-1}): `;
                
                const input = document.createElement('input');
                input.type = 'range';
                input.min = '0';
                input.max = (dim.size - 1).toString();
                input.value = '0';
                input.id = `dim_${dim.name}`;
                input.style.width = '150px';
                
                const valueSpan = document.createElement('span');
                valueSpan.textContent = '0';
                valueSpan.id = `dim_${dim.name}_value`;
                
                input.addEventListener('input', function() {
                    valueSpan.textContent = this.value;
                });
                
                dimControl.appendChild(label);
                dimControl.appendChild(input);
                dimControl.appendChild(valueSpan);
                dimensionControls.appendChild(dimControl);
            }
        });
        
        dimensionSection.style.display = 'block';
    } else {
        dimensionSection.style.display = 'none';
    }
}

// 隐藏维度控制
function hideDimensionControls() {
    document.getElementById('dimensionSection').style.display = 'none';
}

// 获取当前选择的维度参数
function getDimensionParams() {
    const params = new URLSearchParams();
    const dimensionControls = document.getElementById('dimensionControls');
    const inputs = dimensionControls.querySelectorAll('input[type="range"]');
    
    inputs.forEach(input => {
        if (input.id.startsWith('dim_')) {
            const dimName = input.id.substring(4); // 移除 'dim_' 前缀
            params.append(`dim_${dimName}`, input.value);
        }
    });
    
    return params.toString();
}

// 可视化选定的变量
function visualizeVariable() {
    const variableName = document.getElementById('variableSelect').value;
    if (!variableName) {
        alert('请选择一个变量');
        return;
    }
    
    console.log('开始可视化变量:', variableName);
    
    // 显示加载状态
    const visualizeButton = document.getElementById('visualizeVariable');
    const originalText = visualizeButton.textContent;
    visualizeButton.textContent = '加载中...';
    visualizeButton.disabled = true;
    
    // 获取坐标数据
    fetch(`${API_BASE}/coordinates`)
    .then(r => r.json())
    .then(coordData => {
        if (coordData.error) {
            throw new Error(coordData.error);
        }
        
        console.log('坐标数据加载成功');
        
        // 先存储当前变量信息
        currentVariable = {
            name: variableName,
            data: null,
            coordinates: {
                latitudes: coordData.latitudes,
                longitudes: coordData.longitudes
            },
            stats: null
        };
        
        // 显示颜色条控制
        document.getElementById('colorbarSection').style.display = 'block';
        document.getElementById('colorbar').style.display = 'flex';
        
        // 创建3D可视化（只使用图像覆盖模式）
        create3DVisualization(null, coordData.latitudes, coordData.longitudes, null);
        
    })
    .catch(error => {
        console.error('可视化错误:', error);
        alert('可视化失败: ' + error.message);
    })
    .finally(() => {
        visualizeButton.textContent = originalText;
        visualizeButton.disabled = false;
    });
}

// 创建3D可视化
function create3DVisualization(data, latitudes, longitudes, stats) {
    if (!viewer) {
        alert('3D地球未初始化');
        return;
    }
    
    console.log('创建3D可视化，数据点数:', latitudes.length * longitudes.length);
    
    // 清除之前的数据源和图像层
    if (currentDataSource) {
        viewer.dataSources.remove(currentDataSource);
    }
    
    // 移除之前的图像层
    const imageryLayers = viewer.imageryLayers;
    for (let i = imageryLayers.length - 1; i >= 1; i--) {
        imageryLayers.remove(imageryLayers.get(i));
    }
    
    // 只使用灰度图像覆盖模式
    console.log('使用图像覆盖模式显示数据');
    createImageOverlay();
}

// 创建图像覆盖层
function createImageOverlay() {
    if (!currentVariable) {
        console.error('没有当前变量数据');
        return;
    }
    
    const variableName = currentVariable.name;
    
    // 获取维度参数和颜色方案
    const dimensionParams = getDimensionParams();
    const colorScheme = document.getElementById('colorScheme').value || 'viridis';
    
    // 构建API URL
    let apiUrl = `${API_BASE}/image/${variableName}`;
    const params = new URLSearchParams();
    
    if (dimensionParams) {
        const dimParams = new URLSearchParams(dimensionParams);
        for (const [key, value] of dimParams) {
            params.append(key, value);
        }
    }
    
    params.append('colorScheme', colorScheme);
    apiUrl += `?${params.toString()}`;
    
    // 首先调用图像生成API
    fetch(apiUrl)
    .then(response => {
        if (!response.ok) {
            throw new Error(`生成图像失败: ${response.status}`);
        }
        return response.json();
    })
    .then(imageData => {
        console.log('图像生成成功:', imageData);
        
        // 从API响应中获取图像URL和数据范围
        const imageUrl = imageData.image_url;
        const dataMin = imageData.data_min;
        const dataMax = imageData.data_max;
        
        console.log('创建图像覆盖层:', imageUrl);
        console.log('数据范围:', dataMin, '到', dataMax);
        
        // 更新颜色范围显示
        document.getElementById('minValue').value = dataMin.toFixed(3);
        document.getElementById('maxValue').value = dataMax.toFixed(3);
        
        // 创建颜色条
        createColorbar(dataMin, dataMax);
        
        // 获取坐标范围
        const coordinates = currentVariable.coordinates;
        const minLon = Math.min(...coordinates.longitudes);
        const maxLon = Math.max(...coordinates.longitudes);
        const minLat = Math.min(...coordinates.latitudes);
        const maxLat = Math.max(...coordinates.latitudes);
        
        console.log(`坐标范围: 经度 ${minLon} 到 ${maxLon}, 纬度 ${minLat} 到 ${maxLat}`);
        
        // 创建图像提供者
        const imageryProvider = new Cesium.SingleTileImageryProvider({
            url: imageUrl,
            rectangle: Cesium.Rectangle.fromDegrees(minLon, minLat, maxLon, maxLat)
        });
        
        // 添加图像层
        const imageryLayer = viewer.imageryLayers.addImageryProvider(imageryProvider);
        
        // 设置透明度
        const opacity = parseFloat(document.getElementById('opacity').value);
        imageryLayer.alpha = opacity;
        
        // 缩放到数据范围
        const rectangle = Cesium.Rectangle.fromDegrees(minLon, minLat, maxLon, maxLat);
        viewer.camera.setView({
            destination: rectangle
        });
        
        console.log('图像覆盖层创建完成');
    })
    .catch(error => {
        console.error('创建图像覆盖层失败:', error);
        alert('创建图像覆盖层失败: ' + error.message);
    });
}

// 根据数值获取颜色
function getColorForValue(normalizedValue, colorScheme = 'viridis') {
    const colorMaps = {
        viridis: [
            [68, 1, 84],
            [72, 40, 120],
            [62, 74, 137],
            [49, 104, 142],
            [38, 130, 142],
            [31, 158, 137],
            [53, 183, 121],
            [109, 205, 89],
            [180, 222, 44],
            [253, 231, 37]
        ],
        plasma: [
            [13, 8, 135],
            [75, 3, 161],
            [125, 3, 168],
            [168, 34, 150],
            [203, 70, 121],
            [229, 107, 93],
            [248, 148, 65],
            [253, 195, 40],
            [239, 248, 33]
        ],
        coolwarm: [
            [59, 76, 192],
            [98, 130, 234],
            [141, 176, 254],
            [184, 208, 249],
            [221, 221, 221],
            [245, 183, 142],
            [235, 127, 96],
            [215, 48, 39],
            [165, 0, 38]
        ],
        jet: [
            [0, 0, 143],
            [0, 0, 255],
            [0, 127, 255],
            [0, 255, 255],
            [127, 255, 127],
            [255, 255, 0],
            [255, 127, 0],
            [255, 0, 0],
            [127, 0, 0]
        ]
    };
    
    const colors = colorMaps[colorScheme] || colorMaps.viridis;
    
    const index = Math.floor(normalizedValue * (colors.length - 1));
    const nextIndex = Math.min(index + 1, colors.length - 1);
    const t = (normalizedValue * (colors.length - 1)) - index;
    
    const color1 = colors[index];
    const color2 = colors[nextIndex];
    
    const r = Math.round(color1[0] + t * (color2[0] - color1[0]));
    const g = Math.round(color1[1] + t * (color2[1] - color1[1]));
    const b = Math.round(color1[2] + t * (color2[2] - color1[2]));
    
    return `rgb(${r}, ${g}, ${b})`;
}

// 创建颜色条
function createColorbar(minVal, maxVal) {
    // 确保colorbar容器可见
    const colorbarDiv = document.getElementById('colorbar');
    colorbarDiv.style.display = 'block';
    
    colorbarCanvas = document.getElementById('colorbarCanvas');
    if (!colorbarCanvas) {
        console.error('colorbarCanvas元素未找到');
        return;
    }
    colorbarCtx = colorbarCanvas.getContext('2d');
    
    const width = colorbarCanvas.width;
    const height = colorbarCanvas.height;
    
    // 创建灰度渐变
    const gradient = colorbarCtx.createLinearGradient(0, 0, 0, height);
    
    // 添加灰度颜色停止点（从白色到黑色）
    gradient.addColorStop(0, 'rgb(255, 255, 255)'); // 白色（最大值）
    gradient.addColorStop(1, 'rgb(0, 0, 0)');       // 黑色（最小值）
    
    // 绘制颜色条
    colorbarCtx.fillStyle = gradient;
    colorbarCtx.fillRect(0, 0, width, height);
    
    // 更新标签
    updateColorbarLabels(minVal, maxVal);
}

// 更新颜色条标签
function updateColorbarLabels(minVal, maxVal) {
    const labelsDiv = document.getElementById('colorbarLabels');
    labelsDiv.innerHTML = '';
    
    // 创建5个标签
    for (let i = 0; i <= 4; i++) {
        const value = maxVal - (i / 4) * (maxVal - minVal);
        const label = document.createElement('div');
        label.textContent = value.toFixed(3);
        label.style.textAlign = 'left';
        labelsDiv.appendChild(label);
    }
}

// 生成颜色条
function generateColorbar(minValue, maxValue, colorScheme = 'viridis') {
    const colorbarDiv = document.getElementById('colorbar');
    colorbarDiv.innerHTML = '';
    
    // 创建颜色条容器
    const colorbarContainer = document.createElement('div');
    colorbarContainer.style.display = 'flex';
    colorbarContainer.style.flexDirection = 'column';
    colorbarContainer.style.height = '200px';
    colorbarContainer.style.width = '30px';
    colorbarContainer.style.border = '1px solid #ccc';
    
    // 创建颜色渐变
    const steps = 50;
    for (let i = 0; i < steps; i++) {
        const colorDiv = document.createElement('div');
        const ratio = i / (steps - 1);
        const color = getColorForValue(ratio, colorScheme);
        colorDiv.style.backgroundColor = color;
        colorDiv.style.height = `${200/steps}px`;
        colorDiv.style.width = '100%';
        colorbarContainer.appendChild(colorDiv);
    }
    
    // 创建标签容器
    const labelsContainer = document.createElement('div');
    labelsContainer.style.display = 'flex';
    labelsContainer.style.flexDirection = 'column';
    labelsContainer.style.justifyContent = 'space-between';
    labelsContainer.style.height = '200px';
    labelsContainer.style.marginLeft = '5px';
    labelsContainer.style.fontSize = '12px';
    
    // 添加标签
    const numLabels = 5;
    for (let i = 0; i < numLabels; i++) {
        const labelDiv = document.createElement('div');
        const value = maxValue - (i / (numLabels - 1)) * (maxValue - minValue);
        labelDiv.textContent = value.toFixed(2);
        labelsContainer.appendChild(labelDiv);
    }
    
    // 组装颜色条
    const wrapper = document.createElement('div');
    wrapper.style.display = 'flex';
    wrapper.appendChild(colorbarContainer);
    wrapper.appendChild(labelsContainer);
    
    colorbarDiv.appendChild(wrapper);
}

// 更新颜色条
function updateColorbar() {
    const minValue = parseFloat(document.getElementById('minValue').value);
    const maxValue = parseFloat(document.getElementById('maxValue').value);
    const colorScheme = document.getElementById('colorScheme').value;
    
    if (isNaN(minValue) || isNaN(maxValue) || minValue >= maxValue) {
        alert('请输入有效的最小值和最大值（最小值必须小于最大值）');
        return;
    }
    
    // 更新全局变量
    window.colorbarMin = minValue;
    window.colorbarMax = maxValue;
    window.colorScheme = colorScheme;
    
    // 重新生成colorbar
    generateColorbar(minValue, maxValue, colorScheme);
    
    // 如果有当前变量，重新可视化
    if (currentVariable) {
        createImageOverlay(currentVariable);
    }
}

// 事件监听器
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM加载完成，开始初始化...');
    
    // 初始化Cesium
    initializeCesium();
    
    // 检查页面元素
    const loadButton = document.getElementById('loadFile');
    const fileInput = document.getElementById('fileInput');
    
    if (!loadButton || !fileInput) {
        console.error('找不到必要的页面元素');
        alert('页面加载错误：找不到必要的界面元素');
        return;
    }
    
    console.log('页面元素检查完成，添加事件监听器...');
    
    // 文件上传事件
    loadButton.addEventListener('click', function() {
        console.log('加载文件按钮被点击');
        const file = fileInput.files[0];
        
        if (!file) {
            alert('请选择一个NC文件');
            return;
        }
        
        if (!file.name.toLowerCase().endsWith('.nc')) {
            alert('请选择.nc格式的文件');
            return;
        }
        
        uploadNCFile(file);
    });
    
    // 文件选择变化事件
    fileInput.addEventListener('change', function() {
        console.log('文件选择发生变化:', this.files[0]);
    });
    
    // 其他事件监听器
    document.getElementById('visualizeVariable').addEventListener('click', visualizeVariable);
    document.getElementById('updateColorbar').addEventListener('click', updateColorbar);
    
    // 透明度滑块事件
    document.getElementById('opacity').addEventListener('input', function(e) {
        const opacity = parseFloat(e.target.value);
        document.getElementById('opacityValue').textContent = (opacity * 100).toFixed(0) + '%';
        
        // 更新图像层透明度
        const imageryLayers = viewer.imageryLayers;
        for (let i = imageryLayers.length - 1; i >= 1; i--) {
            imageryLayers.get(i).alpha = opacity;
        }
    });
    
    // 颜色方案变化监听
    document.getElementById('colorScheme').addEventListener('change', function() {
        if (currentVariable) {
            updateColorbar();
        }
    });
    
    console.log('事件监听器添加完成');
});