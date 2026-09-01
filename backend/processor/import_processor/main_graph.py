"""知识库导入 LangGraph 工作流：7 节点。"""

import logging

from langgraph.constants import END
from langgraph.graph import StateGraph

from backend.processor.import_processor.nodes.a_node_entry import NodeEntry
from backend.processor.import_processor.nodes.b_node_pdf_to_md import NodePDFToMD
from backend.processor.import_processor.nodes.c_node_md_img import NodeMDImg
from backend.processor.import_processor.nodes.d_node_document_split import NodeDocumentSplit
from backend.processor.import_processor.nodes.e_node_model_tagging import NodeModelTagging
from backend.processor.import_processor.nodes.f_node_bge_embedding import NodeBGEEmbedding
from backend.processor.import_processor.nodes.g_node_import_milvus import NodeImportMilvus
from backend.processor.import_processor.state import ImportGraphState


class KBImportWorkflow:
    """知识库导入工作流。"""

    def __init__(self):
        self.__compiled_graph = None

    @property
    def graph(self):
        if self.__compiled_graph is None:
            self.__compiled_graph = self.build_graph()
        return self.__compiled_graph

    @staticmethod
    def route_after_entry(state: ImportGraphState) -> str:
        if state.get("is_pdf_read_enabled"):
            return "b_node_pdf_to_md"
        if state.get("is_md_read_enabled"):
            return "c_node_md_img"
        logging.info("未启用任何读取方式，直接结束流程")
        return END

    def build_graph(self):
        graph = StateGraph(ImportGraphState)
        graph.add_node("a_node_entry", NodeEntry())
        graph.add_node("b_node_pdf_to_md", NodePDFToMD())
        graph.add_node("c_node_md_img", NodeMDImg())
        graph.add_node("d_node_document_split", NodeDocumentSplit())
        graph.add_node("e_node_model_tagging", NodeModelTagging())
        graph.add_node("f_node_bge_embedding", NodeBGEEmbedding())
        graph.add_node("g_node_import_milvus", NodeImportMilvus())

        graph.set_entry_point("a_node_entry")
        graph.add_conditional_edges(
            "a_node_entry",
            self.route_after_entry,
            {
                "c_node_md_img": "c_node_md_img",
                "b_node_pdf_to_md": "b_node_pdf_to_md",
                END: END,
            },
        )
        graph.add_edge("b_node_pdf_to_md", "c_node_md_img")
        graph.add_edge("c_node_md_img", "d_node_document_split")
        graph.add_edge("d_node_document_split", "e_node_model_tagging")
        graph.add_edge("e_node_model_tagging", "f_node_bge_embedding")
        graph.add_edge("f_node_bge_embedding", "g_node_import_milvus")
        graph.add_edge("g_node_import_milvus", END)
        return graph.compile()

    def run(self, state: ImportGraphState, stream: bool = False):
        if stream:
            return self.graph.stream(state, stream_mode="values")
        return self.graph.invoke(state)
