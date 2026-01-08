import os
import sys

os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import json
import logging
import argparse
import time
from queue import Queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import threading

# 设置路径
SCRIPT_DIR = Path(__file__).parent.absolute()
ANDROID_WORLD_PATH = "../android_world"
if ANDROID_WORLD_PATH not in sys.path:
    sys.path.insert(0, ANDROID_WORLD_PATH)

from trajectory_rollout.client import MtAndroidEnvClient
from trajectory_rollout.agents import LLMEvaluator

# 导入 Agent
from world_model_guided.world_model_guided_agent import WorldModelGuidedAgent
from world_model_guided.ocr_module import OCRModule

# 导入 vLLM 世界模型客户端
from world_model_api.vllm_client import WorldModelVLLMClient

# 导入 sandbox 工具
from sandbox_uitls import IP_POOL, add_sandbox_to_log, load_existing_sandboxes, remove_host_from_log

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.ERROR)

LOCK = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument(
        "--task_family",
        type=str,
        default="android_world",
        choices=["android_world", "android", "information_retrieval", "miniwob", "miniwob_subset"],
    )
    parser.add_argument("--reasoner_model_name", type=str, default="aws.claude-sonnet-4.5")
    parser.add_argument("--actor_model_name", type=str, default="aws.claude-sonnet-4")
    parser.add_argument("--num_threads", type=int, default=min(os.cpu_count() or 8, 8))
    parser.add_argument("--n_task_combinations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    # 世界模型相关（vLLM）
    parser.add_argument("--world_model_url", type=str, default="http://localhost:8410/v1",
                       help="vLLM 服务地址")
    parser.add_argument("--world_model_name", type=str, default="textual-sketch-world-model",
                       help="世界模型名称")
    parser.add_argument(
        "--num_candidate_actions",
        type=int,
        default=1,
        help="候选动作数量",
    )

    parser.add_argument("--wait_after_action_seconds", type=float, default=2.0)
    parser.add_argument("--enable_evaluator", action="store_true", help="开启在线评估(only_final)")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def setup_logging(debug: bool = False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s@%(threadName)s [%(levelname)s] - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def create_world_model_client(url: str, model_name: str) -> WorldModelVLLMClient:
    logging.info(f"创建世界模型客户端: {url}, 模型: {model_name}")
    client = WorldModelVLLMClient(
        base_url=url,
        model_name=model_name,
    )
    
    # 检查服务状态
    if client.health_check():
        logging.info("✓ 世界模型服务可用")
    else:
        logging.warning("⚠ 世界模型服务可能不可用，但仍继续执行")
    
    return client


def worker_thread(args, thread_id, host_ip, host_port, task_queue: Queue, existing_tasks: list, world_model_client):
    client = None
    try:
        cluster_endpoint = os.environ["SANDBOX_CLUSTER_ENDPOINT"]
        application_secret_token = os.environ["SANDBOX_APPLICATION_TOKEN"]

        logging.info(f"线程 {thread_id} 开始执行，使用 IP: {host_ip}:{host_port}")

        client = MtAndroidEnvClient(cluster_endpoint, application_secret_token, host_ip, host_port, False)
        if host_ip is None:
            add_sandbox_to_log(client.host_ip, client.host_port)
        
        logging.debug(f"Reinitializing suite, n_task_combinations={args.n_task_combinations}, seed={args.seed}, task_family={args.task_family}")
        client.reinitialize_suite(n_task_combinations=args.n_task_combinations, seed=args.seed, task_family=args.task_family)
        
        save_path = Path(args.save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        ocr_module = OCRModule()
        
        evaluator = None
        if args.enable_evaluator:
            evaluator = LLMEvaluator("aws.claude-sonnet-4.5", None, only_final=True)
        
        agent = WorldModelGuidedAgent(
            client=client,
            reasoner_model_name=args.reasoner_model_name,
            actor_model_name=args.actor_model_name,
            llm_evaluator=evaluator,
            wait_after_action_seconds=args.wait_after_action_seconds,
            save_path=save_path / f"thread_{thread_id}",
            thread_id=thread_id,
            world_model=world_model_client,
            ocr_module=ocr_module,
            num_candidate_actions=args.num_candidate_actions,
        )

        # 任务队列初始化
        with LOCK:
            if task_queue.empty():
                task_list = []
                for task_name in agent.task_list:
                    task_length = agent.client.get_suite_task_length(task_name)
                    for task_idx in range(task_length):
                        task_list.append((task_name, task_idx))
                random.shuffle(task_list)
                for task_data in task_list:
                    task_queue.put(task_data)
                logging.info(f"Got task queue, {task_queue.qsize()} items in total")
                for _ in range(args.num_threads):
                    task_queue.put(None)

        # 运行 rollout
        agent.run_rollout(task_queue=task_queue, existing_tasks=existing_tasks)
        logging.info(f"线程 {thread_id} 执行完成")
        
    except Exception as e:
        logging.error(f"线程 {thread_id} 执行出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise
    finally:
        if client is not None and client.auto_stop_sandbox:
            remove_host_from_log(client.host_ip, client.host_port)


def main():
    args = parse_args()
    setup_logging(args.debug)

    if args.save_path is None:
        raise ValueError("--save_path is required. e.g. results/xxx/trajectory")

    random.seed(args.seed)
    
    # 创建世界模型客户端
    world_model_client = create_world_model_client(args.world_model_url, args.world_model_name)
    
    # 线程配置
    thread_configs = []
    for i in range(args.num_threads):
        if len(IP_POOL) > 0:
            ip, port = IP_POOL.pop(0)
        else:
            ip, port = None, None
        thread_configs.append((i, ip, port))

    task_queue = Queue()
    
    save_path = Path(args.save_path)
    
    # resume 机制
    existing_tasks = []
    if save_path.exists():
        for thread_path in save_path.glob("thread_*"):
            if not thread_path.is_dir():
                continue
            for traj in thread_path.iterdir():
                reward_path = traj / "reward.json"
                if not reward_path.exists():
                    continue
                with open(reward_path, "r", encoding="utf-8") as f:
                    reward = json.load(f)
                if reward.get("reward", -1) != -1:
                    existing_tasks.append(traj.name.split("-", 2)[-1])
    logging.info(f"Found {len(existing_tasks)} existing trajectories in {save_path.as_posix()}")

    # DEBUG 模式
    if os.environ.get("DEBUG", "0").lower() in ["1", "true"] or args.debug:
        worker_thread(args, 0, thread_configs[0][1], thread_configs[0][2], task_queue, existing_tasks, world_model_client)
        return

    logging.info(f"启动 {args.num_threads} 个线程进行并行执行")
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        # 提交所有任务
        futures = []
        for thread_id, host_ip, host_port in thread_configs:
            future = executor.submit(worker_thread, args, thread_id, host_ip, host_port, task_queue, existing_tasks, world_model_client)
            futures.append(future)

        # 等待所有任务完成
        try:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"任务执行失败: {str(e)}")
        except KeyboardInterrupt:
            logging.info("收到中断信号，正在关闭...")
            executor.shutdown(wait=False)


if __name__ == "__main__":
    load_existing_sandboxes()
    main()


