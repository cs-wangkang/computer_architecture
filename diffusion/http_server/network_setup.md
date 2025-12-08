# 网络配置指南

## 当前网络情况

- **容器内部IP**: 172.17.0.10 (Docker内部网络，无法从外部访问)
- **公网IP**: 36.139.231.213 (需要配置端口映射)
- **服务监听**: 0.0.0.0:5000 (已正确配置)

## 问题分析

172.17.0.10 是 Docker 容器的内部 IP 地址，属于 Docker 的默认网段 (172.17.0.0/16)。这个 IP 只能在容器内部或 Docker 网络中访问，无法从外部直接访问。

## 解决方案

### 方案1: 使用公网IP访问（需要端口映射）

如果服务运行在 Docker 容器中，需要配置端口映射：

```bash
# 如果使用docker run启动，需要添加端口映射
docker run -p 5000:5000 ...

# 或者修改docker-compose.yml
ports:
  - "5000:5000"
```

然后使用公网IP访问：
```
http://36.139.231.213:5000/
```

### 方案2: 检查防火墙设置

确保端口5000已开放：

```bash
# 检查防火墙状态
sudo ufw status
# 或
sudo iptables -L -n | grep 5000

# 开放端口（如果需要）
sudo ufw allow 5000/tcp
# 或
sudo iptables -A INPUT -p tcp --dport 5000 -j ACCEPT
```

### 方案3: 使用SSH隧道（临时方案）

如果无法直接访问，可以使用SSH端口转发：

```bash
# 在本地机器执行
ssh -L 5000:localhost:5000 user@36.139.231.213

# 然后在本地浏览器访问
http://localhost:5000/
```

### 方案4: 使用Nginx反向代理

配置Nginx反向代理，使用标准HTTP端口（80/443）：

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 方案5: 检查云服务商安全组

如果使用云服务器（阿里云、腾讯云等），需要在安全组中开放5000端口：

1. 登录云服务器控制台
2. 找到安全组配置
3. 添加入站规则：端口5000，协议TCP，源0.0.0.0/0

## 验证服务是否可访问

### 从服务器内部测试

```bash
# 测试本地访问
curl http://localhost:5000/health

# 测试127.0.0.1
curl http://127.0.0.1:5000/health

# 测试0.0.0.0
curl http://0.0.0.0:5000/health
```

### 从外部测试

```bash
# 使用公网IP测试
curl http://36.139.231.213:5000/health

# 或使用telnet测试端口是否开放
telnet 36.139.231.213 5000
```

## 推荐配置

1. **确保服务绑定到 0.0.0.0**（已配置 ✓）
2. **配置端口映射**（如果是Docker）
3. **开放防火墙端口**
4. **配置云服务商安全组**
5. **使用Nginx反向代理**（生产环境推荐）

## 快速检查脚本

创建检查脚本 `check_network.sh`:

```bash
#!/bin/bash
echo "=== 网络配置检查 ==="
echo "1. 服务进程:"
ps aux | grep "python.*app.py" | grep -v grep

echo -e "\n2. 端口监听:"
netstat -tlnp 2>/dev/null | grep 5000 || ss -tlnp 2>/dev/null | grep 5000

echo -e "\n3. 本地测试:"
curl -s http://localhost:5000/health || echo "本地访问失败"

echo -e "\n4. 公网IP:"
curl -s ifconfig.me

echo -e "\n5. 防火墙状态:"
sudo ufw status 2>/dev/null || echo "ufw未安装"
```

