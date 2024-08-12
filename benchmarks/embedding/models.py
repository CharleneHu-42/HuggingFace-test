from dataclasses import dataclass
from typing import Optional


@dataclass
class Model:
    name: str
    rev: Optional[str] = None  # Revision ID of model
    remote_code_required: bool = False

    def __str__(self) -> str:
        return f"Model({self.model_str})"

    @property
    def file_str(self) -> str:
        out = self.name.replace("/", "__")
        if self.rev is not None:
            out += "__" + self.rev.replace(".", "-").replace("/", "-")
        return out

    @property
    def model_str(self) -> str:
        model_str = self.name
        if self.rev is not None:
            model_str += f"::{self.rev}"
        if self.remote_code_required:
            model_str += "!!"
        return model_str

    @classmethod
    def from_model_str(cls, model_str: str):
        """
        Model string format: <Model Name>[::<Model Revision>][!!]

        Args:
        - <Model Name>: Model name to import from Transformers library
        - <Model Revision> (optional): Specific revision

        Optional Suffixes:
        - `!!`: Denotes remote code required for model
        """
        components = model_str.split("::")
        model_ref = None
        # Extract remote code required flag
        remote_code_required = components[-1].endswith("!!")
        if remote_code_required:
            tmp = components.pop()[:-2]
            if tmp:
                components.append(tmp)

        model_name = components[0]
        if len(components) > 1:
            model_ref = "::".join(components[1:])
        return cls(model_name, model_ref, remote_code_required)


VALIDATED_MODELS = {
    "tei": [
        Model("Alibaba-NLP/gte-large-en-v1.5", remote_code_required=True),
        Model("BAAI/bge-large-en-v1.5", "refs/pr/5"),
        Model("Salesforce/SFR-Embedding-2_R", remote_code_required=True),
        Model("intfloat/e5-mistral-7b-instruct"),
        Model("jinaai/jina-embeddings-v2-base-en"),
        Model("sentence-transformers/all-MiniLM-L12-v2"),
        Model("sentence-transformers/all-MiniLM-L6-v2"),
        Model("sentence-transformers/all-distilroberta-v1"),
        Model("sentence-transformers/all-mpnet-base-v2"),
        Model("sentence-transformers/distiluse-base-multilingual-cased-v1"),
        Model("sentence-transformers/distiluse-base-multilingual-cased-v2"),
        Model("sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),
        Model("sentence-transformers/multi-qa-distilbert-cos-v1"),
        Model("sentence-transformers/multi-qa-mpnet-base-dot-v1"),
        Model("sentence-transformers/paraphrase-MiniLM-L3-v2"),
        Model("sentence-transformers/paraphrase-albert-small-v2"),
        Model("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        Model("sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
    ]
}
