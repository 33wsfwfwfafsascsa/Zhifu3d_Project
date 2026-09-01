"""Markdown 图片处理节点：VL 摘要 + MinIO 上传 + URL 替换。"""

import base64
import logging
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

from langchain_openai import ChatOpenAI
from minio import Minio
from minio.deleteobjects import DeleteObject

from backend.config.lm_config import lm_config
from backend.config.minio_config import minio_config
from backend.processor.import_processor.base import BaseNode
from backend.processor.import_processor.exceptions import FileProcessingError, StateFieldError
from backend.processor.import_processor.state import ImportGraphState
from backend.utils.minio_utils import get_minio_client


class NodeMDImg(BaseNode):
    """Markdown 图片处理：摘要生成、MinIO 上传、路径替换。"""

    name = "node_md_img"

    def process(self, state: ImportGraphState):
        md_content, md_path_obj, images_dir = self._step_1_get_content(state)
        if not images_dir.exists():
            self.logger.info("无图片文件夹，跳过图片处理")
            return state

        target_images = self._step_2_scan_images(md_content, images_dir)
        if not target_images:
            self.logger.info("未检测到 MD 中引用了图片，跳过图片处理")
            return state

        summaries = self._step_3_generate_summaries(md_path_obj.stem, target_images)
        new_md_content = self._step_4_upload_and_replace(md_path_obj.stem, target_images, summaries, md_content)
        new_md_file_name = self._step_5_backup_new_md_file(state["md_path"], new_md_content)

        state["md_content"] = new_md_content
        state["md_path"] = new_md_file_name
        return state

    def _step_1_get_content(self, state: ImportGraphState) -> Tuple[str, Path, Path]:
        md_path = state.get("md_path")
        if not md_path:
            raise StateFieldError(field_name="md_path", expected_type=str)

        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            raise FileProcessingError(message=f"MD 文件 {md_path_obj.name} 不存在")

        md_content = state["md_content"]
        images_dir = md_path_obj.parent / "images"
        return md_content, md_path_obj, images_dir

    def _step_2_scan_images(self, md_content: str, images_dir: Path) -> List[Tuple[str, str, Tuple[str, str]]]:
        target_images = []
        for image_file in os.listdir(images_dir):
            file_ext = os.path.splitext(image_file)[1].lower()
            if file_ext not in self.config.image_extensions:
                self.logger.warning("图片格式不支持，跳过：%s", image_file)
                continue
            img_path = str(images_dir / image_file)
            context = self._find_image_in_md(md_content, image_file)
            if not context:
                self.logger.warning("图片未在 MD 中引用，跳过：%s", image_file)
                continue
            target_images.append((image_file, img_path, context))
        return target_images

    @staticmethod
    def _find_image_in_md(md_content: str, image_file: str, context_len: int = 100) -> Tuple[str, str] | None:
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")
        match = pattern.search(md_content)
        if not match:
            return None
        start, end = match.span()
        pre_text = md_content[max(0, start - context_len) : start]
        post_text = md_content[end : min(len(md_content), end + context_len)]
        return pre_text, post_text

    def _step_3_generate_summaries(
        self, doc_stem: str, target_images: List[Tuple[str, str, Tuple[str, str]]]
    ) -> Dict[str, str]:
        summaries = {}
        request_deque: Deque[float] = deque()
        for img_file, image_path, context in target_images:
            self._apply_api_rate_limit(request_deque, max_requests=10)
            summaries[img_file] = self._summarize_image(image_path, root_folder=doc_stem, image_content=context)
        return summaries

    @staticmethod
    def _apply_api_rate_limit(request_times: Deque[float], max_requests: int, window_seconds: int = 60) -> None:
        current_time = time.time()
        while request_times and current_time - request_times[0] >= window_seconds:
            request_times.popleft()
        if len(request_times) >= max_requests:
            sleep_duration = window_seconds - (current_time - request_times[0])
            if sleep_duration > 0:
                logging.getLogger().info("触发 API 速率限制，等待 %.2f 秒", sleep_duration)
                time.sleep(sleep_duration)
                current_time = time.time()
                while request_times and current_time - request_times[0] >= window_seconds:
                    request_times.popleft()
        request_times.append(current_time)

    def _summarize_image(self, image_path: str, root_folder: str, image_content: Tuple[str, str]) -> str:
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")

        try:
            chat_model = ChatOpenAI(
                model=lm_config.vl_model,
                api_key=lm_config.api_key,
                base_url=lm_config.base_url,
                temperature=lm_config.llm_temperature,
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f'这是"{root_folder}"文件中的一张图片，图片上文部分为"{image_content[0]}"，'
                                f'下文部分为"{image_content[1]}"，请用中文简要总结这张图片的内容，'
                                "用于 Markdown 图片标题。"
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ]
            response = chat_model.invoke(messages)
            return response.content.strip().replace("\n", "")
        except Exception as exc:
            self.logger.error("图像总结失败：%s，错误 %s", image_path, exc)
            return "图片描述"

    def _step_4_upload_and_replace(
        self,
        doc_stem: str,
        target_images: List[Tuple[str, str, Tuple[str, str]]],
        summaries: Dict[str, str],
        md_content: str,
    ) -> str:
        minio_client = get_minio_client()
        if not minio_client:
            self.logger.warning("MinIO 客户端不可用，跳过图片上传与替换")
            return md_content

        upload_dir = f"{minio_config.img_dir}/{doc_stem}".replace(" ", "")
        self._clean_minio_directory(minio_client, upload_dir)
        urls = self._upload_images_batch(minio_client, upload_dir, target_images)
        image_info = self._merge_summary_and_url(summaries, urls)
        return self._process_md_file(md_content, image_info)

    def _clean_minio_directory(self, minio_client: Minio, prefix: str) -> None:
        try:
            objects_to_delete = minio_client.list_objects(minio_config.bucket_name, prefix=prefix, recursive=True)
            delete_list = [DeleteObject(obj.object_name) for obj in objects_to_delete]
            if delete_list:
                errors = minio_client.remove_objects(minio_config.bucket_name, delete_list)
                for error in errors:
                    self.logger.error("删除失败：%s", error)
        except Exception as exc:
            self.logger.error("清理 MinIO 目录失败：%s", exc)

    def _upload_images_batch(
        self, minio_client: Minio, upload_dir: str, target_images: List[Tuple]
    ) -> Dict[str, str]:
        urls = {}
        for img_file, img_path, _ in target_images:
            object_name = f"{upload_dir}/{img_file}"
            urls[img_file] = self._upload_to_minio(minio_client, img_path, object_name)
        return urls

    def _upload_to_minio(self, minio_client: Minio, local_path: str, object_name: str) -> str | None:
        try:
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=object_name,
                file_path=local_path,
                content_type=f"image/{os.path.splitext(local_path)[1][1:]}",
            )
            base_url = f"http://{minio_config.endpoint}/{minio_config.bucket_name}"
            return f"{base_url}/{object_name}"
        except Exception as exc:
            self.logger.error("图片上传 MinIO 失败：%s，错误：%s", local_path, exc)
            return None

    @staticmethod
    def _merge_summary_and_url(summaries: Dict[str, str], urls: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
        image_info = {}
        for image_file, summary in summaries.items():
            if url := urls.get(image_file):
                image_info[image_file] = (summary, url)
        return image_info

    @staticmethod
    def _process_md_file(md_content: str, image_info: Dict[str, Tuple[str, str]]) -> str:
        for image_file, (summary, new_url) in image_info.items():
            pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")
            md_content = pattern.sub(lambda m: f"![{summary}]({new_url})", md_content)
        return md_content

    @staticmethod
    def _step_5_backup_new_md_file(origin_md_path: str, md_content: str) -> str:
        new_md_file_name = os.path.splitext(origin_md_path)[0] + "_new.md"
        with open(new_md_file_name, "w", encoding="utf-8") as f:
            f.write(md_content)
        return new_md_file_name
