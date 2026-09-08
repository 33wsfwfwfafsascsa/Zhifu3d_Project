"""知识库导入 CLI：python -m backend.processor.import_processor.cli --file <path> [--type ...] [--model ...]"""

import argparse # 命令行参数解析
import uuid # 生成任务 ID
from pathlib import Path

from backend.processor.import_processor.base import setup_logging
from backend.processor.import_processor.catalog import infer_knowledge_type, load_model_catalog
from backend.processor.import_processor.config import KNOWLEDGE_TYPES
from backend.processor.import_processor.main_graph import KBImportWorkflow
from backend.processor.import_processor.state import create_default_state


def main() -> None:
    setup_logging()
    # --- 参数定义 ---
    parser = argparse.ArgumentParser(description="智服3D 知识库导入")
    parser.add_argument("--file", required=True, help="待导入文件（.pdf / .md）")
    # --type 只允许白名单内的值；缺省时按文件所在目录推断
    parser.add_argument("--type", dest="knowledge_type", choices=sorted(KNOWLEDGE_TYPES), default=None,
                        help="知识类型；缺省按 data/raw 目录名推断")
    # --model：文件级机型；省略表示该文件可能涉及多机型，稍后按 chunk 做 LLM 标注
    parser.add_argument("--model", default=None,
                        help="文件级机型；省略表示多机型文件，按 chunk 级 LLM 标注")
    args = parser.parse_args()

    path = Path(args.file).resolve() # 转绝对路径，避免工作目录不一致
    knowledge_type = args.knowledge_type or infer_knowledge_type(path)
    if knowledge_type not in KNOWLEDGE_TYPES:
        raise SystemExit(
            f"--type 必须为 {sorted(KNOWLEDGE_TYPES)}，"
            "或把文件放在 data/raw/{manuals,faq,policy,troubleshooting}/ 下"
        )

    catalog = load_model_catalog()
    if args.model and catalog and args.model not in catalog:
        # 提示但不阻断：文件级机型不在权威目录时仍按原值写入
        print(f"警告：--model={args.model} 不在权威目录中，将按原值写入")

    state = create_default_state(
        task_id=str(uuid.uuid4()), # 每次导入独立任务 ID
        import_file_path=str(path), # 绝对路径
        knowledge_type=knowledge_type, # 必填校验值
        product_model=args.model or "", # 空 => e 节点走 chunk 级标注
        model_catalog=catalog, # 权威目录供归一化
    )
    print(f"开始导入：{path.name}（type={knowledge_type}, model={args.model or 'chunk 级标注'}）")

    workflow = KBImportWorkflow()
    final_state = {}
     # stream_mode="updates"：每个节点完成后产出一个事件 {node_name: state_update}
    for event in workflow.graph.stream(state, stream_mode="updates"):
        for node_name, node_result in event.items():
            # 记录最近一次返回 chunks 的节点结果（通常是 g 入库节点）
            if isinstance(node_result, dict) and node_result.get("chunks") is not None:
                final_state = node_result
            print(f"[{node_name}] 完成")

    chunk_count = len(final_state.get("chunks", []))
    print(f"导入完成：共 {chunk_count} 个 chunk 已入库")


if __name__ == "__main__":
    main()
