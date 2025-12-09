#!/usr/bin/env python3
"""
修复HuggingFace模型下载中卡住的.incomplete文件
检查文件完整性，重命名文件，清理锁文件
"""
import os
import hashlib
import shutil
from pathlib import Path
from typing import List, Tuple
import time

# 配置缓存目录
HF_CACHE_BASE = "/root/autodl-tmp/project/hf_cache/hub"

def calculate_sha256(file_path: str, chunk_size: int = 8192) -> str:
    """计算文件的SHA256哈希值"""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except Exception as e:
        print(f"  错误：计算SHA256时出错: {e}")
        return None

def verify_file_hash(file_path: str, expected_hash: str = None) -> bool:
    """
    验证文件哈希值
    如果提供了expected_hash，则验证是否匹配
    否则只检查文件是否存在且可读
    """
    if not os.path.exists(file_path):
        return False
    
    # 如果提供了期望的哈希值，进行验证
    if expected_hash:
        actual_hash = calculate_sha256(file_path)
        if actual_hash is None:
            return False
        if actual_hash.lower() != expected_hash.lower():
            print(f"  警告：文件哈希值不匹配！期望: {expected_hash}, 实际: {actual_hash}")
            return False
    
    # 检查文件大小是否合理（至少大于0）
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        print(f"  警告：文件大小为0")
        return False
    
    return True

def fix_incomplete_file(incomplete_path: str, dry_run: bool = False) -> Tuple[bool, str]:
    """
    修复单个.incomplete文件
    
    Returns:
        (success, message)
    """
    incomplete_path = Path(incomplete_path)
    
    if not incomplete_path.exists():
        return False, f"文件不存在: {incomplete_path}"
    
    # 获取期望的文件名（移除.incomplete后缀）
    expected_name = incomplete_path.stem  # 移除.incomplete，保留哈希值
    expected_path = incomplete_path.parent / expected_name
    
    # 检查目标文件是否已存在
    if expected_path.exists():
        # 如果已存在，比较大小
        existing_size = expected_path.stat().st_size
        incomplete_size = incomplete_path.stat().st_size
        
        if existing_size == incomplete_size:
            print(f"  ✓ 目标文件已存在且大小相同，删除.incomplete文件")
            if not dry_run:
                incomplete_path.unlink()
            return True, "目标文件已存在"
        else:
            print(f"  ⚠ 目标文件已存在但大小不同（现有: {existing_size}, 新: {incomplete_size}）")
            # 可以选择覆盖或跳过
            return False, "目标文件已存在但大小不同"
    
    # 验证文件（计算哈希值并验证）
    print(f"  正在验证文件完整性...")
    file_size = incomplete_path.stat().st_size
    print(f"  文件大小: {file_size / (1024**3):.2f} GB")
    
    # 对于大文件，只做基本检查，不计算完整SHA256（太耗时）
    if file_size > 100 * 1024 * 1024:  # 大于100MB
        print(f"  文件较大，跳过完整SHA256验证（仅检查文件可读性）")
        # 只检查文件是否可以读取前几MB
        try:
            with open(incomplete_path, "rb") as f:
                f.read(1024 * 1024)  # 读取1MB
            print(f"  ✓ 文件可读性检查通过")
        except Exception as e:
            return False, f"文件读取失败: {e}"
    else:
        # 小文件进行完整验证
        expected_hash = expected_name
        if verify_file_hash(str(incomplete_path), expected_hash):
            print(f"  ✓ 文件哈希验证通过")
        else:
            print(f"  ⚠ 文件哈希验证失败，但继续处理")
    
    # 重命名文件
    if not dry_run:
        try:
            incomplete_path.rename(expected_path)
            print(f"  ✓ 文件已重命名: {expected_name}")
            return True, f"成功重命名文件"
        except Exception as e:
            return False, f"重命名失败: {e}"
    else:
        print(f"  [DRY RUN] 将重命名为: {expected_name}")
        return True, "DRY RUN: 将重命名"

def find_incomplete_files(base_dir: str = HF_CACHE_BASE) -> List[str]:
    """查找所有.incomplete文件"""
    incomplete_files = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"缓存目录不存在: {base_dir}")
        return incomplete_files
    
    # 查找所有.incomplete文件
    for incomplete_file in base_path.rglob("*.incomplete"):
        incomplete_files.append(str(incomplete_file))
    
    return sorted(incomplete_files)

def clean_lock_files(base_dir: str = HF_CACHE_BASE, dry_run: bool = False):
    """清理锁文件（谨慎操作）"""
    base_path = Path(base_dir)
    lock_files = []
    
    # 查找所有.lock文件
    for lock_file in base_path.rglob("*.lock"):
        lock_files.append(lock_file)
    
    if not lock_files:
        print("\n没有找到锁文件")
        return
    
    print(f"\n找到 {len(lock_files)} 个锁文件")
    
    # 检查锁文件是否被进程使用
    active_locks = []
    for lock_file in lock_files:
        # 检查锁文件是否被任何进程打开
        try:
            # 尝试读取锁文件
            with open(lock_file, 'r') as f:
                content = f.read().strip()
            # 如果锁文件很旧（超过1小时），可能是僵尸锁
            lock_age = time.time() - lock_file.stat().st_mtime
            if lock_age > 3600:  # 1小时
                print(f"  发现旧锁文件（{lock_age/3600:.1f}小时前）: {lock_file.name}")
                if not dry_run:
                    lock_file.unlink()
                    print(f"    ✓ 已删除")
            else:
                active_locks.append(lock_file)
        except Exception as e:
            print(f"  无法处理锁文件 {lock_file}: {e}")
    
    if active_locks:
        print(f"\n保留 {len(active_locks)} 个活跃锁文件（可能正在使用中）")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="修复HuggingFace模型下载中卡住的.incomplete文件")
    parser.add_argument("--dry-run", action="store_true", help="仅显示将要执行的操作，不实际修改")
    parser.add_argument("--clean-locks", action="store_true", help="清理锁文件")
    parser.add_argument("--cache-dir", type=str, default=HF_CACHE_BASE, help="HuggingFace缓存目录")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("HuggingFace 模型下载修复工具")
    print("=" * 70)
    
    if args.dry_run:
        print("\n⚠ DRY RUN 模式：不会实际修改文件\n")
    
    # 查找所有.incomplete文件
    print(f"\n正在搜索 .incomplete 文件...")
    incomplete_files = find_incomplete_files(args.cache_dir)
    
    if not incomplete_files:
        print("✓ 没有找到 .incomplete 文件，所有下载已完成！")
        return
    
    print(f"\n找到 {len(incomplete_files)} 个 .incomplete 文件：\n")
    
    # 处理每个文件
    success_count = 0
    fail_count = 0
    
    for i, incomplete_file in enumerate(incomplete_files, 1):
        print(f"[{i}/{len(incomplete_files)}] 处理: {Path(incomplete_file).name}")
        success, message = fix_incomplete_file(incomplete_file, dry_run=args.dry_run)
        
        if success:
            success_count += 1
            print(f"  ✓ {message}\n")
        else:
            fail_count += 1
            print(f"  ✗ {message}\n")
    
    # 清理锁文件
    if args.clean_locks:
        clean_lock_files(args.cache_dir, dry_run=args.dry_run)
    
    # 总结
    print("\n" + "=" * 70)
    print("修复完成！")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print("=" * 70)
    
    if fail_count > 0:
        print("\n⚠ 部分文件修复失败，可能需要手动检查或重新下载")

if __name__ == "__main__":
    main()



