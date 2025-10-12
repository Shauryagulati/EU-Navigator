import argparse
from typing import Optional
from .retrieval import retrieve_and_rerank
from .synthesizer import synthesize

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--module", help="optional module filter",
                    choices=["Equality_Foundations","Data_IP_TDM","AI_Cyber_Gov"])
    args = ap.parse_args()

    docs = retrieve_and_rerank(args.q, args.module)
    ans = synthesize(args.q, docs)
    print("\n=== ANSWER ===\n")
    print(ans)
