
import torch
import json
from attack.gcg_dynamic_ppl import DynamicCSDTFluencyAttacker


def run_targeted_attack(
    attacker,
    s_src: str,
    s_tgt: str,
    init_len: int = 5,
    max_len: int = 40,
    len_step: int = 5,
    steps_per_len: int = 80,
    batch_size: int = 64,
    top_k: int = 64,
    target_sim: float = 0.88,
    print_prefix: str = ""
) -> tuple[str | None, float]:
    print(f"\n{print_prefix}Targeting: {s_tgt[:70]}{'...' if len(s_tgt)>70 else ''}")
    print(f"Benign prompt : {s_src}")

    suffix, sim = attacker.run_dynamic_attack(
        s_src=s_src,
        s_tgt=s_tgt,
        init_len=init_len,
        max_len=max_len,
        len_step=len_step,
        steps_per_len=steps_per_len,
        batch_size=batch_size,
        top_k=top_k,
        target_sim=target_sim,
    )

    if suffix is not None:
        print(f"→ Success @ sim = {sim:.4f}")
        print(f"Suffix: {suffix}")
        print(f"Combined : {s_src} {suffix}")
        return suffix, sim
    else:
        print(f"→ Failed to reach {target_sim:.3f} (best was lower)")
        return None, sim


def main():
    print("Initializing attacker (this may take a while)...")
    attacker = DynamicCSDTFluencyAttacker(
        embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
        lm_model_name="gpt2",
        lambda_ppl=0.04,
        max_lm_length=140,
    )

    # ────────────────────────────────────────────────
    # Load datasets
    # ────────────────────────────────────────────────
    unsafe_path = "../data/gpt5_results_unsafe.jsonl"
    benign_path = "../data/cleaned_qa.jsonl"

    print(f"Loading unsafe from: {unsafe_path}")
    unsafe_insts = []
    with open(unsafe_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                if "instruction" in d:
                    unsafe_insts.append(d["instruction"])
            except:
                pass

    print(f"Loading benign from: {benign_path}")
    benign_qs = []
    with open(benign_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                if "question" in d:
                    benign_qs.append(d["question"])
            except:
                pass

    print(f"Loaded {len(unsafe_insts)} unsafe  |  {len(benign_qs)} benign")

    if not benign_qs or not unsafe_insts:
        print("No data loaded — exiting.")
        return

    # Simple pairing: cycle through benign questions if fewer than unsafe
    n_test = min(5, len(unsafe_insts))   # small number for initial testing
    for i in range(n_test):
        tgt = unsafe_insts[i]
        # Cycle benign prompts
        benign = benign_qs[i % len(benign_qs)]

        run_targeted_attack(
            attacker=attacker,
            s_src=benign,
            s_tgt=tgt,
            init_len=5,
            max_len=45,
            len_step=5,
            steps_per_len=100,
            batch_size=64,
            top_k=64,
            target_sim=0.80,
            print_prefix=f"[Pair {i+1}/{n_test}] "
        )


if __name__ == "__main__":
    main()
