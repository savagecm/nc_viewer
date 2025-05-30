#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D NC文件可视化器 - 本地服务器
用于启动本地HTTP服务器来运行应用
"""

import http.server
import socketserver
import os
import sys
import webbrowser
from pathlib import Path

# 配置
PORT = 8080
HOST = 'localhost'

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """自定义HTTP请求处理器，添加CORS支持"""
    
    def end_headers(self):
        # 添加CORS头部
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        """处理OPTIONS请求"""
        self.send_response(200)
        self.end_headers()
    
    def log_message(self, format, *args):
        """自定义日志格式"""
        print(f"[{self.log_date_time_string()}] {format % args}")

def main():
    """主函数"""
    # 确保在正确的目录中运行
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # 检查必要文件是否存在
    required_files = ['index.html', 'app.js', 'styles.css']
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print(f"错误: 缺少必要文件: {', '.join(missing_files)}")
        print("请确保所有文件都在同一目录中")
        sys.exit(1)
    
    # 创建服务器
    try:
        with socketserver.TCPServer((HOST, PORT), CustomHTTPRequestHandler) as httpd:
            print(f"\n🚀 3D NC文件可视化器服务器启动成功!")
            print(f"📍 服务器地址: http://{HOST}:{PORT}")
            print(f"📁 服务目录: {script_dir}")
            print(f"\n💡 使用说明:")
            print(f"   1. 在浏览器中打开 http://{HOST}:{PORT}")
            print(f"   2. 点击'选择文件'按钮加载NC文件")
            print(f"   3. 选择要可视化的变量")
            print(f"   4. 使用鼠标拖拽、滚轮缩放来操作3D地球")
            print(f"\n⚠️  按 Ctrl+C 停止服务器")
            print(f"{'='*50}")
            
            # 自动打开浏览器
            try:
                webbrowser.open(f'http://{HOST}:{PORT}')
                print(f"✅ 已自动打开浏览器")
            except Exception as e:
                print(f"⚠️  无法自动打开浏览器: {e}")
                print(f"请手动在浏览器中访问: http://{HOST}:{PORT}")
            
            # 启动服务器
            httpd.serve_forever()
            
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ 端口 {PORT} 已被占用")
            print(f"请尝试以下解决方案:")
            print(f"   1. 关闭占用端口的程序")
            print(f"   2. 修改脚本中的PORT变量为其他端口")
            print(f"   3. 等待几分钟后重试")
        else:
            print(f"❌ 服务器启动失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n\n🛑 服务器已停止")
        print(f"感谢使用 3D NC文件可视化器!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 意外错误: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()