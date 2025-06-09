// 全局变量
let viewer;
let currentDataSource;
let currentVariable;
let fileInfo;
let colorbarCanvas;
let colorbarCtx;
let customColormap = null; // 存储自定义颜色映射

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
            document.getElementById('colorbarSection').style.display = 'block';
            
            // 初始化可视化按钮文本
            updateVisualizeButtonText();
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
        const opacity = parseFloat(document.getElementById('opacity').value) || 1.0;
        
        // 检查是否使用自定义颜色映射
        let useCustomColormap = false;
        if (colorScheme === 'custom' && customColormap) {
            useCustomColormap = true;
        } else if (colorScheme === 'custom' && !customColormap) {
            alert('请先上传自定义颜色映射文件');
            return;
        }
        
        // 获取维度选择参数
        const dimensionParams = getDimensionParams();
        
        // 清除之前的数据源和图像层
        viewer.dataSources.removeAll();
        // 清除所有图像层，为NC数据图片让路
        viewer.imageryLayers.removeAll();
        console.log('已清除所有图像层，准备显示NC数据图片');
        
        // 构建图片请求URL
        let imageUrl = `${API_BASE}/visualization/${variableName}/image?color_scheme=${colorScheme}&opacity=${opacity}`;
        if (useCustomColormap) {
            imageUrl += '&custom_colormap=' + encodeURIComponent(JSON.stringify(customColormap));
        }
        if (dimensionParams) {
            imageUrl += '&' + dimensionParams.substring(1);
        }
        
        console.log('请求图片URL:', imageUrl);
        
        // 获取图片数据和分辨率信息
        const imageResponse = await fetch(imageUrl);
        if (!imageResponse.ok) {
            const errorData = await imageResponse.json();
            throw new Error(errorData.error || `HTTP错误 ${imageResponse.status}`);
        }
        
        const imageData = await imageResponse.json();
        console.log('图片数据信息:', {
            width: imageData.width,
            height: imageData.height,
            data_width: imageData.data_width,
            data_height: imageData.data_height,
            longitude_range: imageData.longitude_range,
            latitude_range: imageData.latitude_range
        });
        
        // 使用base64图片数据
        const base64ImageUrl = imageData.image_data;
        
        // 验证base64数据
        if (!base64ImageUrl || !base64ImageUrl.startsWith('data:image/png;base64,')) {
            throw new Error('无效的图片数据格式: ' + (base64ImageUrl ? base64ImageUrl.substring(0, 50) : 'null'));
        }
        
        console.log('Base64图片数据长度:', base64ImageUrl.length);
        console.log('Base64图片数据前缀:', base64ImageUrl.substring(0, 50));
        
        // 直接使用HTTP响应中的latitude_range和longitude_range构建空间范围
        if (!imageData.latitude_range || !imageData.longitude_range) {
            throw new Error('缺少纬度或经度范围数据');
        }
        
        if (!Array.isArray(imageData.latitude_range) || imageData.latitude_range.length !== 2 ||
            !Array.isArray(imageData.longitude_range) || imageData.longitude_range.length !== 2) {
            throw new Error('纬度或经度范围数据格式错误');
        }
        
        // 使用HTTP响应中的latitude_range和longitude_range
        const extent = {
            min_lat: imageData.latitude_range[0],
            max_lat: imageData.latitude_range[1],
            min_lon: imageData.longitude_range[0],
            max_lon: imageData.longitude_range[1]
        };
        
        if (typeof extent.min_lon !== 'number' || 
            typeof extent.min_lat !== 'number' || 
            typeof extent.max_lon !== 'number' || 
            typeof extent.max_lat !== 'number' ||
            isNaN(extent.min_lon) || isNaN(extent.min_lat) || 
            isNaN(extent.max_lon) || isNaN(extent.max_lat)) {
            throw new Error('无效的空间范围数据: ' + JSON.stringify({latitude_range: imageData.latitude_range, longitude_range: imageData.longitude_range}));
        }
        
        console.log('使用HTTP响应的空间范围:', extent);

        // 添加图片图层 - 使用Entity和Rectangle方式
        try {
            // 先预加载图片，确保图片能正确加载
            console.log('开始预加载图片:', base64ImageUrl.substring(0, 50) + '...');
            const imageLoadPromise = new Promise((resolve, reject) => {
                const img = new Image();
                img.crossOrigin = 'anonymous';
                img.onload = () => {
                    console.log('图片预加载成功，尺寸:', img.width, 'x', img.height);
                    resolve(img);
                };
                img.onerror = (error) => {
                    console.error('图片预加载失败:', error);
                    reject(new Error('图片加载失败'));
                };
                img.src = base64ImageUrl;
            });
            
            // 等待图片加载完成
            await imageLoadPromise;
            
            // 创建数据源
            const dataSource = new Cesium.CustomDataSource('ncImageData');
            await viewer.dataSources.add(dataSource);
            // 打印min_lon, min_lat, max_lon, max_lat
            console.log('空间范围:', extent.min_lon, extent.min_lat, extent.max_lon, extent.max_lat);
            // 使用ImageryLayer方式显示图片
            const imageryProvider = new Cesium.SingleTileImageryProvider({
                url: base64ImageUrl,
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
                image_size: `${imageData.width}x${imageData.height}`,
                data_size: `${imageData.data_width}x${imageData.data_height}`,
                longitude_range: imageData.longitude_range,
                latitude_range: imageData.latitude_range,
                spatial_extent: extent,
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
        console.log('Debug: colorScheme =', colorScheme, 'customColormap =', customColormap);
        let colorbarUrl = `${API_BASE}/visualization/${variableName}/colorbar?color_scheme=${colorScheme}`;
        if (colorScheme === 'custom' && customColormap) {
            colorbarUrl += `&custom_colormap=${encodeURIComponent(JSON.stringify(customColormap))}`;
            console.log('Debug: 添加了custom_colormap参数');
        }
        console.log('Debug: colorbarUrl =', colorbarUrl);
        const colorbarResponse = await fetch(colorbarUrl);
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
                    spatial_extent: extent
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
        const opacity = parseFloat(document.getElementById('opacity').value) || 1.0;
        
        viewer.dataSources.getByName('ncData').forEach(dataSource => {
            dataSource.entities.values.forEach(entity => {
                if (entity.point && entity.properties && entity.properties.value) {
                    const value = entity.properties.value.getValue();
                    const color = getColorForValue(value, minValue, maxValue, colorScheme);
                    const cesiumColor = Cesium.Color.fromCssColorString(color);
                    cesiumColor.alpha = opacity;
                    entity.point.color = cesiumColor;
                }
            });
        });
    }
    
    // 移除透明度控制功能
    
    // 颜色方案变化监听
    document.getElementById('colorScheme').addEventListener('change', function() {
        const customSection = document.getElementById('customColormapSection');
        if (this.value === 'custom') {
            customSection.style.display = 'block';
        } else {
            customSection.style.display = 'none';
        }
        
        // 更新可视化按钮文本
        updateVisualizeButtonText();
        
        if (currentVariable) {
            updateColorbar();
        }
    });
    
    // 透明度变化监听
    document.getElementById('opacity').addEventListener('input', function() {
        document.getElementById('opacityValue').textContent = this.value;
        if (currentVariable) {
            updateColorbar();
        }
    });
    
    // 自定义颜色映射文件上传
    document.getElementById('customColormapFile').addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = function(e) {
            try {
                const jsonData = JSON.parse(e.target.result);
                
                // 验证JSON格式
                if (!Array.isArray(jsonData)) {
                    throw new Error('颜色映射必须是数组格式');
                }
                
                for (let i = 0; i < jsonData.length; i++) {
                    const item = jsonData[i];
                    if (!item.hasOwnProperty('position') || !item.hasOwnProperty('color')) {
                        throw new Error('每个颜色映射项必须包含position和color字段');
                    }
                    if (typeof item.position !== 'number' || item.position < 0 || item.position > 1) {
                        throw new Error('position值必须是0-1之间的数字');
                    }
                    if (!Array.isArray(item.color) || item.color.length !== 3) {
                        throw new Error('color必须是包含3个RGB值的数组');
                    }
                    for (let j = 0; j < 3; j++) {
                        if (typeof item.color[j] !== 'number' || item.color[j] < 0 || item.color[j] > 255) {
                            throw new Error('RGB值必须是0-255之间的整数');
                        }
                    }
                }
                
                // 按position排序
                jsonData.sort((a, b) => a.position - b.position);
                
                customColormap = jsonData;
                alert('自定义颜色映射加载成功！');
                
                // 如果当前选择的是自定义颜色映射，重新可视化
                if (document.getElementById('colorScheme').value === 'custom' && currentVariable) {
                    updateColorbar();
                }
                
            } catch (error) {
                alert('颜色映射文件格式错误: ' + error.message);
                customColormap = null;
            }
        };
        reader.readAsText(file);
    });
    
    // 初始化可视化按钮文本
    updateVisualizeButtonText();
    
    console.log('事件监听器添加完成');
});


// 更新可视化按钮文本显示当前颜色方案
function updateVisualizeButtonText() {
    const colorScheme = document.getElementById('colorScheme').value || 'viridis';
    const button = document.getElementById('visualizeVariable');
    
    // 颜色方案名称映射
    const schemeNames = {
        'viridis': 'Viridis',
        'plasma': 'Plasma',
        'inferno': 'Inferno',
        'magma': 'Magma',
        'cividis': 'Cividis',
        'coolwarm': 'Cool-Warm',
        'RdYlBu': 'Red-Yellow-Blue',
        'RdBu': 'Red-Blue',
        'seismic': 'Seismic',
        'jet': 'Jet',
        'hot': 'Hot',
        'cool': 'Cool',
        'spring': 'Spring',
        'summer': 'Summer',
        'autumn': 'Autumn',
        'winter': 'Winter',
        'gray': '灰度',
        'bone': 'Bone',
        'copper': 'Copper',
        'pink': 'Pink',
        'Greys': 'Greys',
        'Blues': 'Blues',
        'Greens': 'Greens',
        'Reds': 'Reds',
        'Oranges': 'Oranges',
        'Purples': 'Purples',
        'BuGn': 'Blue-Green',
        'BuPu': 'Blue-Purple',
        'GnBu': 'Green-Blue',
        'OrRd': 'Orange-Red',
        'PuBu': 'Purple-Blue',
        'PuRd': 'Purple-Red',
        'RdPu': 'Red-Purple',
        'YlGn': 'Yellow-Green',
        'YlOrBr': 'Yellow-Orange-Brown',
        'YlOrRd': 'Yellow-Orange-Red',
        'custom': '自定义颜色映射'
    };
    
    const schemeName = schemeNames[colorScheme] || colorScheme;
    button.textContent = `可视化 (${schemeName})`;
}

// 根据自定义颜色映射获取颜色
function getColorFromCustomColormap(normalizedValue, colormap) {
    if (!colormap || colormap.length === 0) {
        return 'rgb(0, 0, 0)';
    }
    
    // 如果只有一个颜色点
    if (colormap.length === 1) {
        const color = colormap[0].color;
        return `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    }
    
    // 找到对应的颜色区间
    for (let i = 0; i < colormap.length - 1; i++) {
        const current = colormap[i];
        const next = colormap[i + 1];
        
        if (normalizedValue >= current.position && normalizedValue <= next.position) {
            // 线性插值
            const t = (normalizedValue - current.position) / (next.position - current.position);
            const r = Math.round(current.color[0] + t * (next.color[0] - current.color[0]));
            const g = Math.round(current.color[1] + t * (next.color[1] - current.color[1]));
            const b = Math.round(current.color[2] + t * (next.color[2] - current.color[2]));
            return `rgb(${r}, ${g}, ${b})`;
        }
    }
    
    // 如果超出范围，返回最近的颜色
    if (normalizedValue <= colormap[0].position) {
        const color = colormap[0].color;
        return `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    } else {
        const color = colormap[colormap.length - 1].color;
        return `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    }
}

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