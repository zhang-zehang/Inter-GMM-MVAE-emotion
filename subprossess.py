import subprocess
import time


def run_experiment(script_path, times=10, delay=60):
    for i in range(times):
        seed = i  # 使用迭代值作为种子
        print(f"实验开始 {i + 1}/{times}，种子为 {seed}")
        subprocess.Popen(["/bin/python3", script_path, "--seed", str(seed)])
        time.sleep(delay)  # 等待60秒

if __name__ == "__main__":
    script_path = 'main.py'  # 替换为您实验脚本的路径
    run_experiment(script_path, times=10)

