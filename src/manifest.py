import csv
from dataclasses import dataclass
from typing import List

@dataclass
class DocRow:
    doc_id: str
    title: str
    module: str
    txt_path: str
    pdf_path: str

def load_manifest(path: str) -> List[DocRow]:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(DocRow(**row))
    return out
