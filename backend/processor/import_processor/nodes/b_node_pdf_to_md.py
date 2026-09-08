"""PDF 转 Markdown 节点：MinerU 云端解析（vlm）。"""

import logging
import shutil
import subprocess # 调用 curl.exe 兜底下载
import tempfile # curl 下载临时文件
import time
import uuid
import zipfile
from pathlib import Path

import requests

from backend.config.mineru_config import mineru_config
from backend.processor.import_processor.base import BaseNode
from backend.processor.import_processor.exceptions import FileProcessingError, PdfConversionError, StateFieldError
from backend.processor.import_processor.state import ImportGraphState


class NodePDFToMD(BaseNode):
    """PDF 结构化解析：上传 → 轮询 → 下载解压 → 读取 MD。"""

    name = "node_pdf_to_md" # 基类 logger 名：import.node_pdf_to_md

    def process(self, state: ImportGraphState):
        # [1/5] 开始日志：注意用的是全局 logging，而非 self.logger
        logging.info("node_pdf_to_md 节点开始执行...")
        pdf_path_obj, output_dir_obj = self._step_1_validate_paths(state) # 校验 PDF 与输出目录
        zip_url = self._step_2_upload_and_poll(pdf_path_obj) # 上传 + 轮询直到成功
        md_path = self._step_3_download_and_extract(zip_url, output_dir_obj, pdf_path_obj.stem) # 下载解压

        # [2/5] 把转换出的 MD 全文读入内存
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # [3/5] 更新状态：md_path 指向真实转换产物；md_content 供后续节点直接使用
        state["md_path"] = md_path
        state["md_content"] = md_content
        return state # [4/5] LangGraph 节点返回状态更新

    def _step_1_validate_paths(self, state: ImportGraphState):
        """校验 pdf_path / file_dir 是否齐全，目录不存在则创建。"""
        pdf_path = state.get("pdf_path")
        if not pdf_path:
            raise StateFieldError(field_name="pdf_path", expected_type=str)
        file_dir = state.get("file_dir")
        if not file_dir:
            raise StateFieldError(field_name="file_dir", expected_type=str)

        pdf_path_obj = Path(pdf_path)
        output_dir_obj = Path(file_dir)

        if not pdf_path_obj.exists():
            raise FileProcessingError(message=f"PDF 文件 {pdf_path_obj.name} 不存在")
        if not output_dir_obj.exists():
            self.logger.info("输出目录不存在，自动创建：%s", output_dir_obj.absolute())
            output_dir_obj.mkdir(parents=True, exist_ok=True)

        return pdf_path_obj, output_dir_obj

    def _step_2_upload_and_poll(self, pdf_path_obj: Path):
        """向 MinerU 上传 PDF 并轮询任务状态，成功返回 ZIP 下载地址。"""
        token = mineru_config.api_token
        url = f"{mineru_config.base_url}/file-urls/batch" # MinerU 批量文件上传申请接口

        header = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
        data = {"files": [{"name": pdf_path_obj.name}], "model_version": "vlm"}

        # --- 1. 申请预签名上传地址 ---
        response = requests.post(url, headers=header, json=data)
        if response.status_code != 200:
            raise PdfConversionError(message=f"获取上传链接失败：HTTP {response.status_code}")
        result = response.json()
        if result.get("code") != 0:
            raise PdfConversionError(f"获取上传链接失败：{result}")

        signed_url = result["data"]["file_urls"][0]
        batch_id = result["data"]["batch_id"] # 后续轮询用

        # --- 2. 直接 PUT 文件内容 ---
        with open(pdf_path_obj, "rb") as f:
            res_upload = requests.put(signed_url, data=f) # 流式读文件上传（无显式 timeout）
            if res_upload.status_code != 200:
                raise PdfConversionError(f"文件上传失败：HTTP {res_upload.status_code}")
        self.logger.info("文件上传成功")

        # --- 3. 轮询解析结果 ---
        poll_url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
        start_time = time.time()
        timeout_seconds = 600 # 全局超时 10 分钟
        poll_interval = 3 # 每 3 秒查一次
        self.logger.info("开始轮询 MinerU 任务：batch_id=%s，超时 %ss", batch_id, timeout_seconds)

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                raise TimeoutError(f"MinerU 解析超时（>{timeout_seconds}s），batch_id={batch_id}")
            try:
                res_poll = requests.get(url=poll_url, headers=header, timeout=10)
            except Exception as exc:
                # 网络抖动不算失败：记录后继续轮询
                self.logger.warning("轮询网络异常，%ss 后重试：%s", poll_interval, exc)
                time.sleep(poll_interval)
                continue
            if res_poll.status_code != 200:
                raise PdfConversionError(f"轮询失败：HTTP {res_poll.status_code}")

            poll_data = res_poll.json()
            if poll_data.get("code") != 0:
                raise PdfConversionError(f"轮询业务错误：{poll_data}")
            extract_results = poll_data["data"]["extract_result"]
            result_item = extract_results[0] # 本节点只提交 1 个文件，取第一个结果
            data_state = result_item["state"]

            if data_state == "done":
                self.logger.info("MinerU 解析完成，耗时 %ss", int(elapsed))
                return result_item["full_zip_url"]
            if data_state == "failed":
                err_msg = result_item.get("err_msg", "未知错误")
                raise PdfConversionError(f"MinerU 解析失败，batch_id={batch_id}，错误：{err_msg}")
            self.logger.info("解析中... 已耗时 %ss，状态：%s", int(elapsed), data_state)
            time.sleep(poll_interval)

    def _step_3_download_and_extract(self, zip_url: str, output_dir_obj: Path, pdf_stem: str) -> str:
        """下载 ZIP → 保存 → 清理旧目录 → 解压 → 把 full.md 改名为 <stem>.md。"""
        zip_bytes = self._download_zip_with_retry(zip_url) # 双通道下载，返回完整字节

        zip_save_path = output_dir_obj / f"{pdf_stem}_result.zip" # 保留 ZIP 中间产物
        with open(zip_save_path, "wb") as f:
            f.write(zip_bytes)

        extract_target_dir = output_dir_obj / pdf_stem # 解压到 <file_dir>/<stem>/
        if extract_target_dir.exists():
            shutil.rmtree(extract_target_dir) # 重导时清掉上次解压目录
        extract_target_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_save_path, "r") as zip_file_obj:
            zip_file_obj.extractall(extract_target_dir)

        # MinerU 产物约定：Markdown 文件叫 full.md
        target_md_file = extract_target_dir / "full.md"
        new_md_path = target_md_file.with_name(f"{pdf_stem}.md") # 改成便于阅读的名字
        target_md_file.rename(new_md_path)
        return str(new_md_path.absolute())

    def _download_zip_with_retry(self, zip_url: str, max_retries: int = 3) -> bytes:
        """下载 MinerU 结果 ZIP：requests 指数退避重试，失败则 curl（Schannel）兜底。
        MinerU 的 CDN 端点与 Python/OpenSSL 的 TLS 握手在部分网络下会 EOF，
        而 Windows 自带 curl（Schannel TLS 栈）可正常下载，因此做双通道兜底。
        """
        last_error = None
        # --- 通道 1：requests，重试 3 次，间隔 3/6/9 秒 ---
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(zip_url, timeout=120)
                if response.status_code == 200:
                    return response.content
                last_error = RuntimeError(f"HTTP {response.status_code}")
            except Exception as exc:
                last_error = exc
                self.logger.warning("ZIP 下载失败（第 %s 次）：%s，稍后重试", attempt, exc)
                time.sleep(3 * attempt)

        # --- 通道 2：curl.exe（Windows Schannel），请求 5 次重试 ---
        curl = shutil.which("curl.exe") or shutil.which("curl")
        if curl:
            self.logger.warning("requests 下载失败（%s），改用 curl 兜底", last_error)
            tmp_path = Path(tempfile.gettempdir()) / f"mineru_{uuid.uuid4().hex}.zip"
            # -4 IPv4；-L 跟随重定向；--retry 5 自动重试；-sS 只显示错误
            cmd = [curl, "-4", "-L", "--retry", "5", "--retry-all-errors", "-sS", "-o", str(tmp_path), zip_url]
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=600)
                if result.returncode == 0 and tmp_path.exists():
                    data = tmp_path.read_bytes()
                    tmp_path.unlink(missing_ok=True)
                    return data
                raise RuntimeError(result.stderr.decode(errors="ignore")[:200])
            finally:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"ZIP 下载失败：{last_error}")
