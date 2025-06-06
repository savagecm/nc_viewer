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
        
        // 移除默认影像层并添加earth_8k.jpg作为地球纹理
        viewer.imageryLayers.removeAll();
        
        // 加载earth_8k.jpg作为地球纹理
        console.log('正在加载earth_8k.jpg地球纹理...');
        
        const earthImageryProvider = new Cesium.SingleTileImageryProvider({
            url: './cesium/Assets/Textures/earth_8k.jpg',
            rectangle: Cesium.Rectangle.fromDegrees(-180.0, -90.0, 180.0, 90.0)
        });
        
        viewer.imageryLayers.addImageryProvider(earthImageryProvider);
        console.log('earth_8k.jpg地球纹理加载完成');
        
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

// 可视化选定的变量 - 使用后端完全处理的数据
function visualizeVariable() {
    const variableName = document.getElementById('variableSelect').value;
    
    if (!variableName) {
        alert('请选择一个变量');
        return;
    }
    
    console.log('开始可视化变量:', variableName, '使用后端处理的数据');
    
    // 显示加载状态
    const visualizeButton = document.getElementById('visualizeVariable');
    const originalText = visualizeButton.textContent;
    visualizeButton.textContent = '加载中...';
    visualizeButton.disabled = true;
    
    // 使用后端完全处理的数据进行可视化
    visualizeWithBackendData(variableName)
    .finally(() => {
        visualizeButton.textContent = originalText;
        visualizeButton.disabled = false;
    });
}

// 智能可视化数据加载
async function visualizeWithBackendData(variableName) {
    try {
        // 显示加载状态
        document.getElementById('loadingIndicator').style.display = 'block';
        document.getElementById('loadingIndicator').textContent = '正在生成可视化图片...';
        
        // 获取当前设置
        const colorScheme = document.getElementById('colorScheme').value || 'viridis';
        // 移除opacity参数，不再使用透明度功能
        
        // 获取维度选择参数
        const dimensionParams = getDimensionParams();
        
        // 清除之前的数据源和图像层
        viewer.dataSources.removeAll();
        // 清除所有图像层，为NC数据图片让路
        viewer.imageryLayers.removeAll();
        console.log('已清除所有图像层，准备显示NC数据图片');
        
        // 构建图片请求URL（移除width、height和opacity参数，让后端根据NC数据尺寸生成）
        let imageUrl = `${API_BASE}/visualization/${variableName}/image?color_scheme=${colorScheme}`;
        if (dimensionParams) {
            imageUrl += '&' + dimensionParams.substring(1);
        }
        
        console.log('请求图片URL:', imageUrl);
        
        // 获取数据边界信息
        let infoUrl = `${API_BASE}/visualization/${variableName}/info`;
        if (dimensionParams) {
            infoUrl += '?' + dimensionParams.substring(1);
        }
        
        const infoResponse = await fetch(infoUrl);
        if (!infoResponse.ok) {
            const errorData = await infoResponse.json();
            throw new Error(errorData.error || `HTTP错误 ${infoResponse.status}`);
        }
        
        const dataInfo = await infoResponse.json();
        if (!dataInfo.success) {
            throw new Error(dataInfo.error || '获取数据信息失败');
        }
        
        console.log('数据信息:', dataInfo);
        
        // 验证空间范围数据并处理字段名映射
        const rawExtent = dataInfo.spatial_extent;
        if (!rawExtent) {
            throw new Error('缺少空间范围数据');
        }
        
        // 处理字段名映射：后端可能返回lat_max/lat_min/lon_max/lon_min格式
        const extent = {
            min_lon: rawExtent.min_lon || rawExtent.lon_min,
            max_lon: rawExtent.max_lon || rawExtent.lon_max,
            min_lat: rawExtent.min_lat || rawExtent.lat_min,
            max_lat: rawExtent.max_lat || rawExtent.lat_max
        };
        
        if (typeof extent.min_lon !== 'number' || 
            typeof extent.min_lat !== 'number' || 
            typeof extent.max_lon !== 'number' || 
            typeof extent.max_lat !== 'number' ||
            isNaN(extent.min_lon) || isNaN(extent.min_lat) || 
            isNaN(extent.max_lon) || isNaN(extent.max_lat)) {
            throw new Error('无效的空间范围数据: ' + JSON.stringify(rawExtent));
        }
        
        console.log('验证通过的空间范围:', extent);

        // 添加图片图层 - 使用Entity和Rectangle方式
        try {
            // 先预加载图片，确保图片能正确加载
            console.log('开始预加载图片:', imageUrl);
            const imageLoadPromise = new Promise((resolve, reject) => {
                const img = new Image();
                img.crossOrigin = 'anonymous';
                img.onload = () => {
                    console.log('图片预加载成功');
                    resolve(img);
                };
                img.onerror = (error) => {
                    console.error('图片预加载失败:', error);
                    reject(new Error('图片加载失败: ' + imageUrl));
                };
                img.src = imageUrl;
            });
            
            // 等待图片加载完成
            await imageLoadPromise;
            
            // 创建数据源
            const dataSource = new Cesium.CustomDataSource('ncImageData');
            await viewer.dataSources.add(dataSource);
            
            // 使用ImageryLayer方式显示图片
            const imageryProvider = new Cesium.SingleTileImageryProvider({
                url: imageUrl,
                rectangle: Cesium.Rectangle.fromDegrees(
                    extent.min_lon,
                    extent.min_lat,
                    extent.max_lon,
                    extent.max_lat
                )
            });
            
            const imageryLayer = viewer.imageryLayers.addImageryProvider(imageryProvider);
            // 移除透明度设置，使用默认值
            imageryLayer.show = true;
            
            // 确保NC数据图像层在最上层显示
            const layerIndex = viewer.imageryLayers.indexOf(imageryLayer);
            if (layerIndex !== viewer.imageryLayers.length - 1) {
                viewer.imageryLayers.raise(imageryLayer);
                console.log('已将NC数据图像层提升到顶层');
            }
            
            console.log('图片图层添加成功:', {
                url: imageUrl,
                spatial_extent: dataInfo.spatial_extent,
                // 移除透明度设置，使用默认值
                layerIndex: viewer.imageryLayers.indexOf(imageryLayer),
                totalLayers: viewer.imageryLayers.length
            });
            
            // 打印所有图像层信息
            for (let i = 0; i < viewer.imageryLayers.length; i++) {
                const layer = viewer.imageryLayers.get(i);
                console.log(`图像层 ${i}:`, {
                    show: layer.show,
                    alpha: layer.alpha,
                    providerType: layer.imageryProvider.constructor.name
                });
            }
            
            // 强制刷新场景
            viewer.scene.requestRender();
            
        } catch (error) {
            console.error('添加图片图层失败:', error);
            throw new Error('无法添加图片图层: ' + error.message);
        }
        
        // 调整视图到数据边界
        viewer.camera.setView({
            destination: Cesium.Rectangle.fromDegrees(
                extent.min_lon,
                extent.min_lat,
                extent.max_lon,
                extent.max_lat
            )
        });
        
        // 获取颜色条信息
        const colorbarResponse = await fetch(`${API_BASE}/visualization/${variableName}/colorbar?color_scheme=${colorScheme}`);
        if (colorbarResponse.ok) {
            const colorbarData = await colorbarResponse.json();
            if (colorbarData.success) {
                updateColorbarFromBackendData(colorbarData);
                document.getElementById('colorbarSection').style.display = 'block';
                document.getElementById('colorbar').style.display = 'flex';
                
                // 存储当前变量信息
                currentVariable = {
                    name: variableName,
                    stats: colorbarData.data_range,
                    metadata: dataInfo.metadata
                };
            }
        }
        
        console.log('图片可视化完成');
        
    } catch (error) {
        console.error('智能数据可视化错误:', error);
        alert('可视化失败: ' + error.message);
    } finally {
        // 隐藏加载状态
        document.getElementById('loadingIndicator').style.display = 'none';
    }
}

// 删除了不再需要的数据点渲染函数，现在使用图片渲染

// 根据后端数据更新颜色条
function updateColorbarFromBackendData(colorbarData) {
    const colorbarDiv = document.getElementById('colorbar');
    colorbarDiv.innerHTML = '';
    
    const minValue = colorbarData.data_range.min;
    const maxValue = colorbarData.data_range.max;
    
    // 更新输入框的值
    document.getElementById('minValue').value = minValue.toFixed(3);
    document.getElementById('maxValue').value = maxValue.toFixed(3);
    
    // 创建颜色条容器
    const colorbarContainer = document.createElement('div');
    colorbarContainer.style.display = 'flex';
    colorbarContainer.style.flexDirection = 'column';
    colorbarContainer.style.height = '200px';
    colorbarContainer.style.width = '35px';
    colorbarContainer.style.borderRadius = '8px';
    colorbarContainer.style.overflow = 'hidden';
    colorbarContainer.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.3)';
    colorbarContainer.style.border = '1px solid rgba(255, 255, 255, 0.3)';
    
    // 使用后端提供的颜色条数据
    const colors = colorbarData.colorbar;
    const stepHeight = 200 / colors.length;
    
    colors.forEach((colorStep, index) => {
        const colorDiv = document.createElement('div');
        const rgb = colorStep.color;
        colorDiv.style.backgroundColor = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
        colorDiv.style.height = `${stepHeight}px`;
        colorDiv.style.width = '100%';
        colorbarContainer.appendChild(colorDiv);
    });
    
    // 创建标签容器
    const labelsContainer = document.createElement('div');
    labelsContainer.style.display = 'flex';
    labelsContainer.style.flexDirection = 'column';
    labelsContainer.style.justifyContent = 'space-between';
    labelsContainer.style.height = '200px';
    labelsContainer.style.marginLeft = '8px';
    labelsContainer.style.fontSize = '12px';
    labelsContainer.style.color = '#ffffff';
    labelsContainer.style.fontWeight = '500';
    labelsContainer.style.position = 'relative';
    
    // 添加标签（从最小值到最大值，对应颜色条从上到下）
    const numLabels = 5;
    for (let i = 0; i < numLabels; i++) {
        const labelDiv = document.createElement('div');
        const value = minValue + (i / (numLabels - 1)) * (maxValue - minValue);
        labelDiv.textContent = value.toFixed(2);
        labelDiv.style.position = 'absolute';
        labelDiv.style.top = `${(i / (numLabels - 1)) * 100}%`;
        labelDiv.style.transform = 'translateY(-50%)';
        labelDiv.style.color = '#ffffff';
        labelDiv.style.fontWeight = '500';
        labelDiv.style.textShadow = '1px 1px 2px rgba(0,0,0,0.8)';
        labelsContainer.appendChild(labelDiv);
    }
    
    // 组装颜色条
    const wrapper = document.createElement('div');
    wrapper.style.display = 'flex';
    wrapper.appendChild(colorbarContainer);
    wrapper.appendChild(labelsContainer);
    
    colorbarDiv.appendChild(wrapper);
}





// 根据数值获取颜色 - 使用简化的颜色映射
function getColorForValue(normalizedValue, colorScheme = 'viridis') {
    // 基本颜色映射，用于前端预览
    const basicColorMaps = {
        viridis: [[68, 1, 84], [253, 231, 37]], // 0->深紫色, 1->亮黄色
        plasma: [[13, 8, 135], [239, 248, 33]],
        inferno: [[0, 0, 4], [252, 255, 164]],
        magma: [[0, 0, 4], [252, 253, 191]],
        cividis: [[0, 32, 77], [255, 233, 69]],
        coolwarm: [[59, 76, 192], [165, 0, 38]],
        RdYlBu: [[165, 0, 38], [49, 54, 149]],
        RdBu: [[103, 0, 31], [5, 48, 97]],
        seismic: [[0, 0, 76], [128, 0, 0]],
        jet: [[0, 0, 143], [127, 0, 0]],
        hot: [[0, 0, 0], [255, 255, 255]],
        cool: [[0, 255, 255], [255, 0, 255]],
        spring: [[255, 0, 255], [255, 255, 0]],
        summer: [[0, 128, 102], [255, 255, 102]],
        autumn: [[255, 0, 0], [255, 255, 0]],
        winter: [[0, 0, 255], [0, 255, 128]],
        gray: [[0, 0, 0], [255, 255, 255]],
        bone: [[0, 0, 0], [255, 255, 255]],
        copper: [[0, 0, 0], [255, 200, 127]],
        pink: [[30, 0, 0], [255, 255, 255]],
        Greys: [[0, 0, 0], [255, 255, 255]],
        Blues: [[247, 251, 255], [8, 48, 107]],
        Greens: [[247, 252, 245], [0, 68, 27]],
        Reds: [[255, 245, 240], [103, 0, 13]],
        Oranges: [[255, 245, 235], [127, 39, 4]],
        Purples: [[252, 251, 253], [63, 0, 125]]
    };
    
    const colors = basicColorMaps[colorScheme] || basicColorMaps.viridis;
    
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
    
    // 创建5个标签，从最小值到最大值（顶部到底部）
    for (let i = 0; i <= 4; i++) {
        const value = minVal + (i / 4) * (maxVal - minVal);
        const label = document.createElement('div');
        label.textContent = value.toFixed(3);
        label.style.textAlign = 'left';
        label.style.position = 'absolute';
        label.style.top = `${(i / 4) * 100}%`;
        label.style.transform = 'translateY(-50%)';
        label.style.color = '#ffffff';
        label.style.fontWeight = '500';
        label.style.textShadow = '1px 1px 2px rgba(0,0,0,0.8)';
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
    colorbarContainer.style.width = '35px';
    colorbarContainer.style.borderRadius = '8px';
    colorbarContainer.style.overflow = 'hidden';
    colorbarContainer.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.3)';
    colorbarContainer.style.border = '1px solid rgba(255, 255, 255, 0.3)';
    
    // 创建颜色渐变（从上到下：最大值到最小值）
    const steps = 50;
    for (let i = 0; i < steps; i++) {
        const colorDiv = document.createElement('div');
        const ratio = 1 - (i / (steps - 1)); // 反转ratio，使顶部对应最大值
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
    labelsContainer.style.marginLeft = '8px';
    labelsContainer.style.fontSize = '12px';
    labelsContainer.style.color = '#ffffff';
    labelsContainer.style.fontWeight = '500';
    labelsContainer.style.position = 'relative';
    
    // 添加标签（从最小值到最大值，对应颜色条从上到下）
    const numLabels = 5;
    for (let i = 0; i < numLabels; i++) {
        const labelDiv = document.createElement('div');
        const value = minValue + (i / (numLabels - 1)) * (maxValue - minValue);
        labelDiv.textContent = value.toFixed(2);
        labelDiv.style.position = 'absolute';
        labelDiv.style.top = `${(i / (numLabels - 1)) * 100}%`;
        labelDiv.style.transform = 'translateY(-50%)';
        labelDiv.style.color = '#ffffff';
        labelDiv.style.fontWeight = '500';
        labelDiv.style.textShadow = '1px 1px 2px rgba(0,0,0,0.8)';
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
        visualizeVariable();
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
    
    // 更新颜色条按钮事件
     document.getElementById('updateColorbar').addEventListener('click', function() {
         if (!currentVariable || !currentVariable.name) {
             alert('请先选择并可视化一个变量');
             return;
         }
         
         const minValue = parseFloat(document.getElementById('minValue').value);
         const maxValue = parseFloat(document.getElementById('maxValue').value);
         
         if (isNaN(minValue) || isNaN(maxValue) || minValue >= maxValue) {
             alert('请输入有效的最小值和最大值（最小值必须小于最大值）');
             return;
         }
         
         // 更新颜色条
         createColorbar(minValue, maxValue);
         
         // 重新渲染Cesium数据点
         updateCesiumDataColors(minValue, maxValue);
     });
    
    // 更新Cesium数据点颜色
    function updateCesiumDataColors(minValue, maxValue) {
        const colorScheme = document.getElementById('colorScheme').value || 'viridis';
        // 移除opacity参数，不再使用透明度功能
        
        viewer.dataSources.getByName('ncData').forEach(dataSource => {
            dataSource.entities.values.forEach(entity => {
                if (entity.point && entity.properties && entity.properties.value) {
                    const value = entity.properties.value.getValue();
                    const color = getColorForValue(value, minValue, maxValue, colorScheme);
                    entity.point.color = Cesium.Color.fromCssColorString(color);
                }
            });
        });
    }
    
    // 移除透明度控制功能
    
    // 颜色方案变化监听
    document.getElementById('colorScheme').addEventListener('change', function() {
        if (currentVariable) {
            updateColorbar();
        }
    });
    
    console.log('事件监听器添加完成');
});


// 使用GeoJSON渲染数据
async function visualizeWithGeoJSON(variableName) {
    console.log('开始使用GeoJSON渲染变量:', variableName);
    
    // 获取维度选择参数
    const dimensionParams = getDimensionParams();
    
    // 构建请求URL
    let url = `${API_BASE}/geojson/${variableName}`;
    if (dimensionParams) {
        url += dimensionParams;
    }
    
    // 显示加载状态
    document.getElementById('loadingIndicator').style.display = 'block';
    
    try {
        // 获取GeoJSON数据
        const response = await fetch(url);
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP错误 ${response.status}`);
        }
        
        const geojsonData = await response.json();
        console.log('GeoJSON数据获取成功，特征数量:', geojsonData.properties.total_features);
        
        // 清除之前的数据源和图像层
        viewer.dataSources.removeAll();
        // 清除所有图像层，为NC数据图片让路
        viewer.imageryLayers.removeAll();
        console.log('已清除所有图像层，准备显示NC数据图片');
        
        // 获取颜色方案和透明度
        const colorScheme = document.getElementById('colorScheme').value || 'viridis';
        // 移除opacity参数，不再使用透明度功能
        
        // 创建数据源
        const dataSource = new Cesium.GeoJsonDataSource('ncData');
        
        // 设置数据范围
        const minValue = geojsonData.properties.data_min;
        const maxValue = geojsonData.properties.data_max;
        
        // 更新颜色条
        document.getElementById('minValue').value = minValue.toFixed(3);
        document.getElementById('maxValue').value = maxValue.toFixed(3);
        generateColorbar(minValue, maxValue, colorScheme);
        
        // 显示颜色条控制
        document.getElementById('colorbarSection').style.display = 'block';
        document.getElementById('colorbar').style.display = 'flex';
        
        // 加载GeoJSON数据
        console.log('开始加载GeoJSON数据到Cesium...');
        const startTime = performance.now();
        
        // 设置点样式回调函数
        const styleOptions = {
            markerColor: function(feature) {
                const value = feature.properties.value;
                // 归一化值
                const normalizedValue = (value - minValue) / (maxValue - minValue);
                // 获取颜色
                return Cesium.Color.fromCssColorString(getColorForValue(normalizedValue, colorScheme));
            },
            markerSize: 8,
            clampToGround: false
        };
        
        // 加载GeoJSON数据
        await dataSource.load(geojsonData, styleOptions);
        
        // 添加数据源到viewer
        await viewer.dataSources.add(dataSource);
        
        // 缩放到数据范围
        viewer.zoomTo(dataSource);
        
        const endTime = performance.now();
        console.log(`GeoJSON渲染完成，耗时: ${(endTime - startTime) / 1000} 秒`);
        console.log(`渲染了 ${geojsonData.properties.total_features} 个特征`);
        
    } catch (error) {
        console.error('GeoJSON渲染错误:', error);
        alert('GeoJSON渲染失败: ' + error.message);
    } finally {
        // 隐藏加载状态
        document.getElementById('loadingIndicator').style.display = 'none';
    }
}