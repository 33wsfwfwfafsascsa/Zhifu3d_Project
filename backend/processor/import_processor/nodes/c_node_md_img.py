"""Markdown 图片处理节点：VL 摘要 + MinIO 上传 + URL 替换。"""

import base64
import logging
import os
import re
import time
from collections import deque # 限流窗口队列
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
        # [1/7] 读取 MD 内容/路径，推导 images/ 目录
        md_content, md_path_obj, images_dir = self._step_1_get_content(state)

        if not images_dir.exists():
            # 目录不存在 = 文档没有图片：直接放行到 d 节点
            self.logger.info("无图片文件夹，跳过图片处理")
            return state

        # [2/7] 找出“受支持格式且被 MD 引用”的图片
        target_images = self._step_2_scan_images(md_content, images_dir)
        if not target_images:
            self.logger.info("未检测到 MD 中引用了图片，跳过图片处理")
            return state

        # [3/7] 逐张调用 VL 模型生成中文摘要（限流 10 次/分钟）
        summaries = self._step_3_generate_summaries(md_path_obj.stem, target_images)
        # [4/7] 上传 MinIO 并重写 MD（MinIO 不可用则原样返回）
        new_md_content = self._step_4_upload_and_replace(md_path_obj.stem, target_images, summaries, md_content)
        # [5/7] 把处理后的 MD 落盘为 <原名>_new.md
        new_md_file_name = self._step_5_backup_new_md_file(state["md_path"], new_md_content)

        # [6/7] 状态指向新内容与新路径
        state["md_content"] = new_md_content
        state["md_path"] = new_md_file_name
        return state # [7/7]

    def _step_1_get_content(self, state: ImportGraphState) -> Tuple[str, Path, Path]:
        """读取输入：md_path 必须存在，md_content 直接取自 state。"""
        md_path = state.get("md_path")
        if not md_path:
            raise StateFieldError(field_name="md_path", expected_type=str)

        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            raise FileProcessingError(message=f"MD 文件 {md_path_obj.name} 不存在")

        md_content = state["md_content"] # a 节点已全文读入 / b 节点转换产物
        images_dir = md_path_obj.parent / "images" # MinerU/人工目录约定：图片放 images/
        return md_content, md_path_obj, images_dir

    def _step_2_scan_images(self, md_content: str, images_dir: Path) -> List[Tuple[str, str, Tuple[str, str]]]:
        """扫描图片目录，返回 (文件名, 本地路径, 上下文(前文,后文))。"""
        target_images = []
        for image_file in os.listdir(images_dir):
            file_ext = os.path.splitext(image_file)[1].lower()
            if file_ext not in self.config.image_extensions:
                self.logger.warning("图片格式不支持，跳过：%s", image_file)
                continue
            img_path = str(images_dir / image_file)
            context = self._find_image_in_md(md_content, image_file) # 必须被正文引用
            if not context:
                self.logger.warning("图片未在 MD 中引用，跳过：%s", image_file)
                continue
            target_images.append((image_file, img_path, context))
        return target_images

    @staticmethod
    def _find_image_in_md(md_content: str, image_file: str, context_len: int = 100) -> Tuple[str, str] | None:
        """
        查找MD内容中指定图片的所有引用位置，并返回每个位置的上下文文本
        :param md_content: MD文件完整内容
        :param image_file: 图片文件名（含后缀）
        :param context_len: 上下文截取长度，默认前后各100字符
        :return: 每个图片的(上文, 下文)元组，无匹配则返回None
        """
        # 1、定义正则表达式
        # ![描述](images/文件名.扩展名)
        # r"字符串"：不要将其中的特殊符号进行转义
        # re.escape 转义图片文件名中的特殊字符，避免正则语法错误
        # .* 贪婪匹配 .*? 非贪婪匹配
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
        """
        批量为待处理图片生成内容摘要，带API速率限制防止触发大模型限流
        :param doc_stem: 文档文件名（不含后缀），作为大模型prompt上下文
        :param targets: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
        :param requests_per_minute: 每分钟最大API请求数，默认9次（按大模型限制调整）
        :return: 图片摘要字典，键：图片文件名，值：图片内容摘要
        """
        summaries = {}
        # 外部初始化双端队列，用于API速率限制，跨循环复用
        request_deque: Deque[float] = deque()

        # 循环处理图片
        for img_file, image_path, context in target_images:
            # 速率限制
            self._apply_api_rate_limit(request_deque, max_requests=10)
            # 调用大模型生成图片摘要
            summaries[img_file] = self._summarize_image(image_path, root_folder=doc_stem, image_content=context)
        return summaries

    @staticmethod
    def _apply_api_rate_limit(request_times: Deque[float], max_requests: int, window_seconds: int = 60) -> None:
        """
        通用滑动窗口API速率限制器（抽离为公共工具）
        核心逻辑：维护请求时间戳双端队列，窗口内请求数超上限则自动等待，防止触发第三方API限流
        :param request_times: 存储请求时间戳的双端队列，需外部初始化（全局/单例），跨调用复用
        :param max_requests: 速率限制窗口内的最大允许请求次数
        :param window_seconds: 速率限制滑动窗口时长，默认60秒（1分钟）
        :return: None，超出限制时会阻塞等待
        """
        current_time = time.time()
        # 1. 清理滑动窗口外的过期请求时间戳，保证队列仅存窗口内的请求
        while request_times and current_time - request_times[0] >= window_seconds:
            request_times.popleft()
        # 2. 窗口内请求数达上限，计算并阻塞等待剩余时间
        if len(request_times) >= max_requests:
            # 计算需要等待的时长（窗口总时长 - 最早请求已存在的时长）
            sleep_duration = window_seconds - (current_time - request_times[0])
            if sleep_duration > 0:
                logging.getLogger().info("触发 API 速率限制，等待 %.2f 秒", sleep_duration)
                time.sleep(sleep_duration)
                # 等待后更新当前时间，重新清理过期请求（避免等待期间有请求过期）
                current_time = time.time()
                while request_times and current_time - request_times[0] >= window_seconds:
                    request_times.popleft()
        request_times.append(current_time) # 记录本次请求

    def _summarize_image(self, image_path: str, root_folder: str, image_content: Tuple[str, str]) -> str:
        """
           调用多模态大模型总结图片内容。

           参数：
           - image_path: 图片本地路径。
           - root_folder: 文档所属文件夹名（提供更多上下文）。
           - image_content: 图片在文档中的上下文 (前文, 后文)。
        """
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
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},# 注意：无论原图是什么格式，这里都声明为 image/jpeg
                    ],
                }
            ]
            response = chat_model.invoke(messages)
            # 摘要压成单行，避免破坏 Markdown 图片 alt 语法
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
        """上传图片到 MinIO 并重写 MD；无 MinIO 则跳过替换。"""
        minio_client = get_minio_client()
        if not minio_client:
            self.logger.warning("MinIO 客户端不可用，跳过图片上传与替换")
            return md_content

        # 构造上传目录，去除文件名中的空格
        upload_dir = f"{minio_config.img_dir}/{doc_stem}".replace(" ", "")
        self._clean_minio_directory(minio_client, upload_dir) # 幂等重导：先清旧图
        urls = self._upload_images_batch(minio_client, upload_dir, target_images)
        # 合并图片摘要和URL，过滤上传失败的图片
        image_info = self._merge_summary_and_url(summaries, urls)
        # 替换MD内容中的本地图片引用为MinIO远程引用
        return self._process_md_file(md_content, image_info)

    def _clean_minio_directory(self, minio_client: Minio, prefix: str) -> None:
        """
        幂等性清理：上传前先删除 MinIO 中指定目录下的旧文件。
        防止重名文件导致的内容混淆或垃圾堆积。
        """
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
        """
        批量上传待处理图片至MinIO，返回图片文件名与访问URL的映射关系
        """
        urls = {}
        for img_file, img_path, _ in target_images:
            object_name = f"{upload_dir}/{img_file}"
            urls[img_file] = self._upload_to_minio(minio_client, img_path, object_name)
        return urls

    def _upload_to_minio(self, minio_client: Minio, local_path: str, object_name: str) -> str | None:
        """
        将单张本地图片上传至MinIO对象存储，并返回公网可访问URL
        """
        try:
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=object_name,
                file_path=local_path,
                content_type=f"image/{os.path.splitext(local_path)[1][1:]}",
            )
            # 构造MinIO基础访问URL
            base_url = f"http://{minio_config.endpoint}/{minio_config.bucket_name}"
            return f"{base_url}/{object_name}"
        except Exception as exc:
            self.logger.error("图片上传 MinIO 失败：%s，错误：%s", local_path, exc)
            return None

    @staticmethod
    def _merge_summary_and_url(summaries: Dict[str, str], urls: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
        """合并摘要与 URL：仅保留“有 URL 的图片”，生成 {文件名: (摘要, URL)}。"""
        image_info = {}
        for image_file, summary in summaries.items():
            if url := urls.get(image_file):
                image_info[image_file] = (summary, url)
        return image_info

    @staticmethod
    def _process_md_file(md_content: str, image_info: Dict[str, Tuple[str, str]]) -> str:
        """
        核心功能：替换MD内容中的本地图片引用为MinIO远程引用
        替换规则：![原描述](本地路径) → ![图片摘要](MinIO访问URL)
        """
        # 遍历 image_info 字典的每一项：key=图片文件名，value=(摘要, 新URL)
        for image_file, (summary, new_url) in image_info.items():
            pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")
            md_content = pattern.sub(lambda m: f"![{summary}]({new_url})", md_content)
        return md_content

    @staticmethod
    def _step_5_backup_new_md_file(origin_md_path: str, md_content: str) -> str:
        """把处理结果写为 *_new.md（保留原文件，便于人工 diff）。"""
        new_md_file_name = os.path.splitext(origin_md_path)[0] + "_new.md"
        with open(new_md_file_name, "w", encoding="utf-8") as f:
            f.write(md_content)
        return new_md_file_name
