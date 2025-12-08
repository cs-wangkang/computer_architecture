#!/bin/bash
# 网络访问修复脚本

echo "=========================================="
echo "网络访问修复工具"
echo "=========================================="

# 检查是否在Docker容器中
if [ -f /.dockerenv ] || grep -q docker /proc/self/cgroup 2>/dev/null; then
    echo "检测到运行在Docker容器中"
    echo "需要配置端口映射，请使用以下命令启动容器："
    echo ""
    echo "docker run -p 5000:5000 ..."
    echo ""
    echo "或者修改docker-compose.yml添加："
    echo "ports:"
    echo "  - \"5000:5000\""
    echo ""
fi

# 检查iptables规则
echo -e "\n检查iptables规则:"
if command -v iptables &> /dev/null; then
    echo "当前5000端口规则:"
    sudo iptables -L -n | grep 5000 || echo "  未找到5000端口规则"
    
    read -p "是否添加iptables规则开放5000端口? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo iptables -A INPUT -p tcp --dport 5000 -j ACCEPT
        echo "✓ 已添加iptables规则"
    fi
fi

# 检查ufw防火墙
if command -v ufw &> /dev/null; then
    echo -e "\n检查ufw防火墙:"
    sudo ufw status | grep 5000 || echo "  5000端口未在ufw中开放"
    
    read -p "是否使用ufw开放5000端口? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo ufw allow 5000/tcp
        echo "✓ 已开放5000端口"
    fi
fi

# 提供SSH隧道方案
echo -e "\n=========================================="
echo "临时访问方案 - SSH隧道:"
echo "=========================================="
echo "在本地机器执行以下命令："
echo ""
echo "ssh -L 5000:localhost:5000 root@36.139.231.213"
echo ""
echo "然后在本地浏览器访问: http://localhost:5000/"
echo ""

# 提供Nginx反向代理建议
echo "=========================================="
echo "生产环境建议 - Nginx反向代理:"
echo "=========================================="
echo "配置Nginx使用80/443端口，然后转发到5000端口"
echo "这样可以避免直接暴露5000端口"
echo ""

echo "=========================================="
echo "云服务商安全组配置:"
echo "=========================================="
echo "如果使用云服务器，需要在安全组中开放5000端口："
echo "1. 登录云服务器控制台"
echo "2. 找到安全组/防火墙配置"
echo "3. 添加入站规则："
echo "   - 端口: 5000"
echo "   - 协议: TCP"
echo "   - 源: 0.0.0.0/0 (或指定IP)"
echo ""

