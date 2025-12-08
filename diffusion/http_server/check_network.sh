#!/bin/bash
# 网络配置检查脚本

echo "=========================================="
echo "网络配置检查"
echo "=========================================="

echo -e "\n1. 服务进程状态:"
ps aux | grep "python.*app.py" | grep -v grep || echo "  服务未运行"

echo -e "\n2. 端口监听状态:"
if command -v netstat &> /dev/null; then
    netstat -tlnp 2>/dev/null | grep ":5000" || echo "  端口5000未监听"
elif command -v ss &> /dev/null; then
    ss -tlnp 2>/dev/null | grep ":5000" || echo "  端口5000未监听"
else
    echo "  无法检查端口状态（netstat/ss不可用）"
fi

echo -e "\n3. 本地访问测试:"
if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "  ✓ 本地访问成功"
    curl -s http://localhost:5000/health | python3 -m json.tool 2>/dev/null || curl -s http://localhost:5000/health
else
    echo "  ✗ 本地访问失败"
fi

echo -e "\n4. 网络接口IP:"
hostname -I

echo -e "\n5. 公网IP:"
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || curl -s ipinfo.io/ip 2>/dev/null)
if [ -n "$PUBLIC_IP" ]; then
    echo "  $PUBLIC_IP"
    echo -e "\n6. 公网访问测试:"
    if curl -s --connect-timeout 3 http://$PUBLIC_IP:5000/health > /dev/null 2>&1; then
        echo "  ✓ 公网访问成功"
    else
        echo "  ✗ 公网访问失败（可能是防火墙/安全组未开放）"
    fi
else
    echo "  无法获取公网IP"
fi

echo -e "\n7. 防火墙状态:"
if command -v ufw &> /dev/null; then
    sudo ufw status | head -5
elif command -v firewall-cmd &> /dev/null; then
    sudo firewall-cmd --list-ports 2>/dev/null || echo "  无法检查防火墙"
else
    echo "  防火墙工具不可用"
fi

echo -e "\n=========================================="
echo "访问地址:"
echo "  本地: http://localhost:5000/"
if [ -n "$PUBLIC_IP" ]; then
    echo "  公网: http://$PUBLIC_IP:5000/"
fi
echo "=========================================="

