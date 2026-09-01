"""知识库导入 CLI：python -m backend.processor.import_processor.cli --file <path> [--type ...] [--model ...]"""

import argparse
import logging
import uuid
from pathlib import Path

from backend.processor.import_processor.base import setup_logging
from backend.processor.import_processor.config import KNOWLEDGE_TYPES
from backend.processor.import_processor.main_graph import KBImportWorkflow
from backend.processor.import_processor.state import create_default_state


def infer_knowledge_type(path: Path) -> str:
    mapping = {
        "manuals": "manual",
        "faq": "faq",
        "policy": "policy",
        "troubleshooting": "troubleshooting",
    }
    return mapping.get(path.parent.name.lower(), "")


def load_model_catalog() -> list[str]:
    try:
        from backend.mock_business_api.config import get_connection

        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT model FROM products ORDER BY model")
                rows = cur.fetchall()
        finally:
            conn.close()
        return [row["model"] for row in rows]
    except Exception as exc:
        logging.getLogger(__name__).warning("读取机型目录失败（%s），标注将不做归一化", exc)
        return []


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="智服3D 知识库导入")
    parser.add_argument("--file", required=True, help="待导入文件（.pdf / .md）")
    parser.add_argument("--type", dest="knowledge_type", choices=sorted(KNOWLEDGE_TYPES), default=None,
                        help="知识类型；缺省按 data/raw 目录名推断")
    parser.add_argument("--model", default=None,
                        help="文件级机型；省略表示多机型文件，按 chunk 级 LLM 标注")
    args = parser.parse_args()

    path = Path(args.file).resolve()
    knowledge_type = args.knowledge_type or infer_knowledge_type(path)
    if knowledge_type not in KNOWLEDGE_TYPES:
        raise SystemExit(
            f"--type 必须为 {sorted(KNOWLEDGE_TYPES)}，"
            "或把文件放在 data/raw/{manuals,faq,policy,troubleshooting}/ 下"
        )

    catalog = load_model_catalog()
    if args.model and catalog and args.model not in catalog:
        print(f"警告：--model={args.model} 不在权威目录中，将按原值写入")

    state = create_default_state(
        task_id=str(uuid.uuid4()),
        import_file_path=str(path),
        knowledge_type=knowledge_type,
        product_model=args.model or "",
        model_catalog=catalog,
    )
    print(f"开始导入：{path.name}（type={knowledge_type}, model={args.model or 'chunk 级标注'}）")

    workflow = KBImportWorkflow()
    final_state = {}
    for event in workflow.graph.stream(state, stream_mode="updates"):
        for node_name, node_result in event.items():
            if isinstance(node_result, dict) and node_result.get("chunks") is not None:
                final_state = node_result
            print(f"[{node_name}] 完成")

    chunk_count = len(final_state.get("chunks", []))
    print(f"导入完成：共 {chunk_count} 个 chunk 已入库")


if __name__ == "__main__":
    main()
