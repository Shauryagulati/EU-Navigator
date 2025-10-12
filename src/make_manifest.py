import csv
from pathlib import Path

#Paths
TXT_DIR = Path("data/txt")
PDF_DIR = Path("data/pdf")
OUT_CSV = Path("data/eu_plp_manifest.csv")

#Mapping for modules based on DocID
def infer_module(name: str) -> str:
    name = name.lower()
    if name.startswith("oj_l_2024"):
        return "AI_Cyber_Gov"
    elif any(k in name for k in ["32016r0679", "32019l0790", "32009l0024"]):
        return "Data_IP_TDM"
    else:
        return "Equality_Foundations"

def make_manifest():
    rows = []
    for txt_file in TXT_DIR.glob("*.txt"):
        base = txt_file.stem
        pdf_guess = PDF_DIR / f"{base}.pdf"
        module = infer_module(base)
        rows.append({
            "doc_id": base.replace("_EN_TXT", "").replace(" - EN", "").replace(" ", "_"),
            "title": base.replace("_EN_TXT", "").replace("_", " "),
            "module": module,
            "txt_path": str(txt_file),
            "pdf_path": str(pdf_guess if pdf_guess.exists() else ""),
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "title", "module", "txt_path", "pdf_path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[manifest] Saved {len(rows)} rows → {OUT_CSV}")

if __name__ == "__main__":
    make_manifest()
