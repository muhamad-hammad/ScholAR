import sys
from types import SimpleNamespace


def _make_langgraph():
    class StateGraph:
        def __init__(self, schema=None):
            self.schema = schema
            self._nodes = {}
            self._entry = None
            self._cond = None
            self._cond_map = {}
            self._edges = []

        def add_node(self, name, fn):
            self._nodes[name] = fn

        def set_entry_point(self, name):
            self._entry = name

        def add_conditional_edges(self, from_node, decision_fn, mapping):
            self._cond = decision_fn
            self._cond_map = mapping

        def add_edge(self, a, b):
            self._edges.append((a, b))

        def compile(self):
            nodes = self._nodes
            cond = self._cond
            cond_map = self._cond_map
            entry = self._entry
            edges = list(self._edges)

            class Compiled:
                def run(self, state):
                    # simple execution: run entry node then follow conditional mapping
                    cur = entry
                    # run router
                    state = nodes[cur](state)
                    # decide next node
                    next_name = cond(state) if cond is not None else None
                    # If the decision function returned a mapping key (e.g. 'QNA'/'SUMMARY'), map it
                    if next_name in cond_map:
                        next_name = cond_map[next_name]

                    if next_name in nodes:
                        # run the decided node (retrieval or summarization)
                        state = nodes[next_name](state)

                        # If there's an edge from retrieval to generation, run generation
                        if (next_name, "generation_node") in edges and "generation_node" in nodes:
                            state = nodes["generation_node"](state)

                        # If there's an edge from summarization to END, the summarization node
                        # should have already set final_answer; nothing further required here.

                    # compile should return final state as dict-like
                    return state

            return Compiled()

    return SimpleNamespace(StateGraph=StateGraph, END="END")


# Inject fake langgraph.graph
if "langgraph.graph" not in sys.modules:
    sys.modules["langgraph.graph"] = _make_langgraph()


# Minimal fake for transformers to allow imports during tests
if "transformers" not in sys.modules:
    mod = SimpleNamespace()

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model, use_fast=True):
            return SimpleNamespace(model=model)

    def _pipeline(task, model=None, tokenizer=None, framework=None, **kwargs):
        def _call(prompt):
            return [{"generated_text": f"{prompt[:50]}"}]

        return _call

    class _TFAutoModelForCausalLM:
        pass

    mod.AutoTokenizer = _AutoTokenizer
    mod.pipeline = _pipeline
    mod.TFAutoModelForCausalLM = _TFAutoModelForCausalLM
    sys.modules["transformers"] = mod


# Minimal langchain_huggingface placeholders
if "langchain_huggingface.llms" not in sys.modules:
    sys.modules["langchain_huggingface.llms"] = SimpleNamespace(HuggingFacePipeline=lambda **k: SimpleNamespace(pipeline=k.get("pipeline")))
if "langchain_huggingface.embeddings" not in sys.modules:
    sys.modules["langchain_huggingface.embeddings"] = SimpleNamespace(HuggingFaceEmbeddings=lambda **k: SimpleNamespace(model_name=k.get("model_name")))


# Minimal langchain_community placeholders (document loaders, vectorstores)
if "langchain_community" not in sys.modules:
    lc_comm = SimpleNamespace()
    # simple loader placeholders
    class _DedocFileLoader:
        def __init__(self, path, with_tables=False):
            self.path = path

        def load(self):
            return []

    class _UnstructuredPDFLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            return []

    class _PyMuPDFLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            return []

    class _PyPDFLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            return []

    class _Chroma:
        @staticmethod
        def from_documents(documents=None, embedding=None, persist_directory=None):
            return SimpleNamespace(persist=lambda: None, similarity_search=lambda q, k=4: [])

    doc_mod = SimpleNamespace(
        DedocFileLoader=_DedocFileLoader,
        UnstructuredPDFLoader=_UnstructuredPDFLoader,
        PyMuPDFLoader=_PyMuPDFLoader,
        PyPDFLoader=_PyPDFLoader,
    )
    vs_mod = SimpleNamespace(Chroma=_Chroma)
    sys.modules["langchain_community"] = lc_comm
    sys.modules["langchain_community.document_loaders"] = doc_mod
    sys.modules["langchain_community.vectorstores"] = vs_mod


# Minimal langchain_text_splitters placeholder to avoid importing the installed package
if "langchain_text_splitters" not in sys.modules:
    class _TokenTextSplitter:
        def __init__(self, chunk_size=1024, chunk_overlap=128):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        @staticmethod
        def from_huggingface_tokenizer(tokenizer=None, chunk_size=1024, chunk_overlap=128):
            return _TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        def split_text(self, text: str):
            # naive splitter for tests: return the whole text as single chunk
            return [text]

        def split_documents(self, docs):
            out = []
            for d in docs:
                text = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
                out.append(type(d)(text, **(getattr(d, "metadata", {}) or {})) if hasattr(type(d), '__init__') else d)
            return docs

    class _RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=1024, chunk_overlap=128):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_text(self, text: str):
            return [text]

    sys.modules["langchain_text_splitters"] = SimpleNamespace(TokenTextSplitter=_TokenTextSplitter, RecursiveCharacterTextSplitter=_RecursiveCharacterTextSplitter)


# Minimal langchain_core placeholders
if "langchain_core" not in sys.modules:
    lc_prompts = SimpleNamespace(ChatPromptTemplate=lambda *a, **k: None)
    lc_runnables = SimpleNamespace(Runnable=object)
    lc_messages = SimpleNamespace(BaseMessage=object, AIMessage=object, HumanMessage=object, SystemMessage=object)
    lc_documents = SimpleNamespace(Document=object)
    sys.modules["langchain_core"] = SimpleNamespace(prompts=lc_prompts, runnables=lc_runnables, messages=lc_messages, documents=lc_documents)
    # Also set the submodules for direct imports
    sys.modules["langchain_core.prompts"] = lc_prompts
    sys.modules["langchain_core.runnables"] = lc_runnables
    sys.modules["langchain_core.messages"] = lc_messages
    sys.modules["langchain_core.documents"] = lc_documents
    sys.modules["langchain_core.embeddings"] = SimpleNamespace(Embeddings=object)
