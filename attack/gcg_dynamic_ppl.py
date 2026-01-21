import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import math
from huggingface_hub import constants
constants.HUGGINGFACE_CO_URL_HOME = "https://hf-mirror.com"
print(constants._HF_DEFAULT_ENDPOINT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DynamicCSDTFluencyAttacker:
    """
    Constraint-based Semantic Drift Targeting
    with Fluency Regularization and Dynamic Suffix Length
    """

    def __init__(
        self,
        embed_model_name: str,
        lm_model_name: str = "gpt2",
        lambda_ppl: float = 0.05,
        max_lm_length: int = 128,
    ):
        # Embedding encoder
        self.embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
        self.embed_model = AutoModel.from_pretrained(embed_model_name).to(device)
        self.embed_model.eval()

        # Language model for fluency
        self.lm_tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
        self.lm_model = AutoModelForCausalLM.from_pretrained(lm_model_name).to(device)
        self.lm_model.eval()

        if self.lm_tokenizer.pad_token is None:
            self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token

        self.lambda_ppl = lambda_ppl
        self.max_lm_length = max_lm_length

        self.embedding_weight = self.embed_model.get_input_embeddings().weight
        self.vocab_size = self.embedding_weight.size(0)

    # ------------------------------------------------
    # Utilities
    # ------------------------------------------------

    def get_embedding(self, input_ids, attention_mask):
        outputs = self.embed_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return F.normalize(cls_emb, dim=1)

    @torch.no_grad()
    def compute_ppl(self, text: str) -> float:
        inputs = self.lm_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_lm_length,
        ).to(device)

        outputs = self.lm_model(**inputs, labels=inputs["input_ids"])
        return math.exp(outputs.loss.item())

    # ------------------------------------------------
    # Fixed-length optimization
    # ------------------------------------------------

    def optimize_fixed_length(
        self,
        s_src: str,
        s_tgt: str,
        suffix_len: int,
        steps: int,
        batch_size: int,
        top_k: int,
        target_sim: float,
    ):
        # Initialize random suffix
        suffix_ids = torch.randint(
            0, self.vocab_size, (suffix_len,), device=device
        )

        # Target embedding
        tgt_inputs = self.embed_tokenizer(
            s_tgt, return_tensors="pt", truncation=True
        ).to(device)

        with torch.no_grad():
            tgt_emb = self.get_embedding(
                tgt_inputs["input_ids"], tgt_inputs["attention_mask"]
            )

        src_ids = self.embed_tokenizer(
            s_src, return_tensors="pt", add_special_tokens=False
        ).input_ids[0].to(device)

        cls_id = torch.tensor([self.embed_tokenizer.cls_token_id], device=device)
        sep_id = torch.tensor([self.embed_tokenizer.sep_token_id], device=device)

        best_sim = -1.0

        for _ in tqdm(range(steps), desc=f"Optimize L={suffix_len}"):
            # Forward pass
            input_ids = torch.cat([cls_id, src_ids, suffix_ids, sep_id]).unsqueeze(0)
            input_embeds = self.embed_model.get_input_embeddings()(input_ids).detach()
            input_embeds.requires_grad_(True)

            outputs = self.embed_model(inputs_embeds=input_embeds)
            cur_emb = F.normalize(outputs.last_hidden_state[:, 0, :], dim=1)

            sim = torch.sum(cur_emb * tgt_emb)
            loss = 1.0 - sim
            loss.backward()

            # Gradient-based token candidates
            start = 1 + len(src_ids)
            grads = input_embeds.grad[0, start : start + suffix_len]
            scores = -torch.matmul(grads, self.embedding_weight.T)
            topk_ids = torch.topk(scores, top_k, dim=1).indices

            # Batch evaluation
            best_candidate = suffix_ids
            best_candidate_score = best_sim

            for _ in range(batch_size):
                cand = suffix_ids.clone()
                pos = torch.randint(0, suffix_len, (1,)).item()
                tok = topk_ids[pos, torch.randint(0, top_k, (1,)).item()]
                cand[pos] = tok

                full_text = s_src + " " + self.embed_tokenizer.decode(
                    cand, skip_special_tokens=True
                )

                # Embedding similarity
                emb_inputs = self.embed_tokenizer(
                    full_text, return_tensors="pt", truncation=True
                ).to(device)

                with torch.no_grad():
                    emb = self.get_embedding(
                        emb_inputs["input_ids"], emb_inputs["attention_mask"]
                    )
                    sim_val = torch.sum(emb * tgt_emb).item()

                # Fluency penalty
                ppl = self.compute_ppl(full_text)
                score = sim_val - self.lambda_ppl * math.log(ppl)
                print(f"Score: {score}, PPL: {ppl}, Sim: {sim_val}")

                if score > best_candidate_score:
                    best_candidate = cand
                    best_candidate_score = score

            suffix_ids = best_candidate
            best_sim = best_candidate_score

            if best_sim >= target_sim:
                break

        return suffix_ids, best_sim

    # ------------------------------------------------
    # Dynamic-length attack
    # ------------------------------------------------

    def run_dynamic_attack(
        self,
        s_src: str,
        s_tgt: str,
        init_len: int = 5,
        max_len: int = 40,
        len_step: int = 5,
        steps_per_len: int = 100,
        batch_size: int = 64,
        top_k: int = 64,
        target_sim: float = 0.9,
    ):
        for L in range(init_len, max_len + 1, len_step):
            suffix, sim = self.optimize_fixed_length(
                s_src,
                s_tgt,
                suffix_len=L,
                steps=steps_per_len,
                batch_size=batch_size,
                top_k=top_k,
                target_sim=target_sim,
            )

            if sim >= target_sim:
                return (
                    self.embed_tokenizer.decode(suffix, skip_special_tokens=True),
                    sim,
                )

        return None, -1.0

# Example usage:
if __name__ == "__main__":
    attacker = DynamicCSDTFluencyAttacker(embed_model_name="sentence-transformers/all-MiniLM-L6-v2")

    source_sentence = "The quick brown fox jumps over the lazy dog."
    target_sentence = "A fast dark-colored fox leaps above a sleepy canine."

    poisoned_suffix, final_sim = attacker.run_dynamic_attack(
        s_src=source_sentence,
        s_tgt=target_sentence,
        init_len=5,
        max_len=30,
        len_step=5,
        steps_per_len=50,
        batch_size=32,
        top_k=32,
        target_sim=0.85,
    )

    if poisoned_suffix:
        print(f"Poisoned Suffix: {poisoned_suffix}")
        print(f"Final Similarity: {final_sim}")
    else:
        print("Failed to find a suitable poisoned suffix.")