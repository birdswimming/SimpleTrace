import argparse
import os


def profile_enable():
    fifo_path = "/tmp/simpletrace"

    # 创建命名管道，如果已存在且不是管道则报错
    if not os.path.exists(fifo_path):
        os.mkfifo(fifo_path)

    parser = argparse.ArgumentParser(description="生成 profile_config.yaml 配置文件")
    parser.add_argument(
        "--mode", type=int, default=0, help="0:both 1:train 2:dataloader"
    )
    parser.add_argument("--ranks", type=str, default="0", help='示例: "0,1,2"')
    parser.add_argument("--schedule-wait", type=int, default=1)
    parser.add_argument("--schedule-active", type=int, default=1)
    parser.add_argument("--relative", type=int, default=1)

    args = parser.parse_args()

    output_file = "/tmp/profile_config.yaml"

    # 写入 YAML 文件
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"mode: {args.mode} # 0:both 1:train 2:dataloader\n")
        f.write(f"ranks: [{args.ranks}] # [0,1,2,4,8]\n")
        f.write(f"schedule_wait: {args.schedule_wait}\n")
        f.write(f"schedule_active: {args.schedule_active}\n")
        f.write(f"relative: {str(bool(args.relative)).lower()}\n")

    # 写入 /tmp/simpletrace 文件
    with open(fifo_path, "w", encoding="utf-8") as f:
        f.write(output_file)
