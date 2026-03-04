import math
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

import lm_eval
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance

from config import Config, IGNORE_INDEX
from transformers import AutoTokenizer

from data import MQARDataset, load_lm_dataset
from kernels.loss import fused_cross_entropy_loss

# IEEE 754: exp(log(DBL_MAX)) == DBL_MAX. Exact overflow cap.
_LOG_EXP_MAX = math.log(sys.float_info.max)


def _sweep_ratios(d_state: int) -> tuple[float, ...]:
    """Generate capacity-cliff sweep ratios from d_state.

    Point count = log2(d_state), density adapts to model scale.
    Range: 2^(-1) (half-capacity) to log2(d_state) (well beyond capacity).
    """
    n = round(math.log2(d_state))
    upper = math.log2(d_state)
    log_lo = math.log10(0.5)
    log_hi = math.log10(upper)
    ratios = [10 ** (log_lo + i * (log_hi - log_lo) / (n - 1)) for i in range(n)]
    return tuple(ratios)


def _eval_grid(d_state: int) -> tuple[int, ...]:
    """Powers of 2 from 2^2 up to 2^(log2(d_state)+1).

    Range and count both derived from d_state.
    """
    upper_exp = round(math.log2(d_state)) + 1
    return tuple(1 << k for k in range(2, upper_exp + 1))


def mqar_accuracy_loop(
    model: nn.Module,
    loader: DataLoader,
    vocab_size: int,
) -> float:
    accuracy = MulticlassAccuracy(
        num_classes=vocab_size,
        ignore_index=IGNORE_INDEX,
        average="micro",
    ).to("cuda")

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
            logits = model(batch["src_ids"], batch["tgt_ids"][:, :-1])
            labels = batch["labels"]
            accuracy.update(logits.view(-1, logits.size(-1)), labels.reshape(-1))

    return accuracy.compute().item()


def evaluate(
    model: nn.Module,
    *,
    num_pairs: int,
    vocab_size: int,
    seed: int,
    num_samples: int,
    batch_size: int,
) -> dict:
    dataset = MQARDataset(
        num_pairs,
        num_pairs,
        num_samples,
        "test",
        vocab_size=vocab_size,
        seed=seed,
    )
    loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    acc = mqar_accuracy_loop(model, loader, vocab_size)
    return {"token_accuracy": acc, "num_pairs": num_pairs}


def _aggregate_seeds(
    model: nn.Module,
    *,
    num_pairs: int,
    vocab_size: int,
    base_seed: int,
    num_samples: int,
    batch_size: int,
    num_eval_seeds: int,
) -> dict:
    accs = [
        evaluate(
            model,
            num_pairs=num_pairs,
            vocab_size=vocab_size,
            seed=base_seed + i,
            num_samples=num_samples,
            batch_size=batch_size,
        )["token_accuracy"]
        for i in range(num_eval_seeds)
    ]
    t = torch.tensor(accs)
    return {
        "token_accuracy_mean": t.mean().item(),
        "token_accuracy_std": t.std().item(),
        "token_accuracy_seeds": accs,
        "num_pairs": num_pairs,
    }


def capacity_cliff(model: nn.Module, config: Config) -> dict:
    d_state = config.d_state
    sweep = [int(d_state * r) for r in _sweep_ratios(d_state)]
    results = []

    for np_ in sweep:
        r = _aggregate_seeds(
            model,
            num_pairs=np_,
            vocab_size=config.vocab_size,
            base_seed=config.seed,
            num_samples=config.eval_num_samples,
            batch_size=config.eval_batch_size,
            num_eval_seeds=config.num_eval_seeds,
        )
        results.append(
            {
                "num_pairs": np_,
                "token_accuracy_mean": r["token_accuracy_mean"],
                "token_accuracy_std": r["token_accuracy_std"],
                "above_capacity": np_ > d_state,
            }
        )

    cliff = None
    max_drop = 0.0
    for i in range(1, len(results)):
        drop = results[i - 1]["token_accuracy_mean"] - results[i]["token_accuracy_mean"]
        if drop > max_drop:
            max_drop = drop
            cliff = results[i]["num_pairs"]

    return {
        "results": results,
        "cliff_point": cliff,
        "max_drop": max_drop,
        "d_state": d_state,
    }


def multi_config_grid(model: nn.Module, config: Config) -> dict:
    grid = [int(x) for x in config.eval_grid.split(",")] if config.eval_grid else list(_eval_grid(config.d_state))
    results = []

    for np_ in grid:
        r = _aggregate_seeds(
            model,
            num_pairs=np_,
            vocab_size=config.vocab_size,
            base_seed=config.seed,
            num_samples=config.eval_num_samples,
            batch_size=config.eval_batch_size,
            num_eval_seeds=config.num_eval_seeds,
        )
        results.append(
            {
                "num_pairs": np_,
                "token_accuracy_mean": r["token_accuracy_mean"],
                "token_accuracy_std": r["token_accuracy_std"],
            }
        )

    accs = [r["token_accuracy_mean"] for r in results]
    uniform_avg = sum(accs) / len(accs)
    return {"results": results, "uniform_avg_accuracy": uniform_avg}


def perplexity_loop(
    model: nn.Module,
    loader: DataLoader,
) -> tuple[float, int]:
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
            logits = model(None, batch["input_ids"][:, :-1])
            labels = batch["labels"][:, 1:]
            loss = fused_cross_entropy_loss(logits, labels, smoothing=0.0, ignore_index=IGNORE_INDEX)
            n = (labels != IGNORE_INDEX).sum().item()
            total_loss += loss.item() * n
            total_tokens += n

    return total_loss, total_tokens


def evaluate_perplexity(model: nn.Module, config: Config) -> dict:
    dataset = load_lm_dataset(config, "test")
    loader = DataLoader(dataset, batch_size=config.eval_batch_size, pin_memory=True)

    total_loss, total_tokens = perplexity_loop(model, loader)
    avg_loss = total_loss / total_tokens
    return {
        "perplexity": math.exp(min(avg_loss, _LOG_EXP_MAX)),
        "avg_cross_entropy": avg_loss,
        "total_tokens": total_tokens,
    }


def evaluate_harness(model: nn.Module, config: Config) -> dict:
    class SlotMambaLM(LM):
        def __init__(self, model, config):
            super().__init__()
            self._model = model
            self._config = config
            self._device = "cuda"
            self._tokenizer = AutoTokenizer.from_pretrained(config.lm_tokenizer)

        @property
        def eot_token_id(self):
            return self._tokenizer.eos_token_id

        @property
        def max_length(self):
            return self._config.lm_seq_length

        @property
        def max_gen_toks(self):
            return self._config.max_gen_toks

        @property
        def batch_size(self):
            return self._config.eval_batch_size

        @property
        def device(self):
            return self._device

        def tok_encode(self, string: str) -> list[int]:
            return self._tokenizer.encode(string, add_special_tokens=False)

        def tok_decode(self, tokens: list[int]) -> str:
            return self._tokenizer.decode(tokens)

        def _model_call(self, inps):
            with torch.no_grad():
                return self._model(None, inps)

        def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
            results = []
            for req in requests:
                context, continuation = req.args
                ctx_enc = self.tok_encode(context)
                cont_enc = self.tok_encode(continuation)
                inp = torch.tensor([ctx_enc + cont_enc], device=self.device, dtype=torch.long)
                inp = inp[:, -self.max_length :]
                logits = self._model_call(inp[:, :-1])
                logits = torch.log_softmax(logits.float(), dim=-1)
                cont_len = len(cont_enc)
                cont_logits = logits[:, -cont_len:, :]
                cont_toks = inp[:, -cont_len:]
                log_probs = torch.gather(cont_logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)
                total_ll = log_probs.sum().item()
                is_greedy = (cont_logits.argmax(dim=-1) == cont_toks).all().item()
                results.append((total_ll, is_greedy))
            return results

        def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float]]:
            results = []
            for req in requests:
                (text,) = req.args
                enc = self.tok_encode(text)
                inp = torch.tensor([enc], device=self.device, dtype=torch.long)
                inp = inp[:, -self.max_length :]
                logits = self._model_call(inp[:, :-1])
                logits = torch.log_softmax(logits.float(), dim=-1)
                toks = inp[:, 1:]
                log_probs = torch.gather(logits, 2, toks.unsqueeze(-1)).squeeze(-1)
                results.append((log_probs.sum().item(),))
            return results

        def generate_until(self, requests: list[Instance]) -> list[str]:
            results = []
            for req in requests:
                context, gen_kwargs = req.args
                max_gen = gen_kwargs["max_gen_toks"]
                stop = gen_kwargs["until"]
                enc = self.tok_encode(context)
                inp = enc[:]
                for _ in range(max_gen):
                    t = torch.tensor([inp[-self.max_length :]], device=self.device, dtype=torch.long)
                    logits = self._model_call(t)
                    next_tok = logits[:, -1, :].argmax(dim=-1).item()
                    inp.append(next_tok)
                    decoded = self.tok_decode(inp[len(enc) :])
                    if any(s in decoded for s in stop):
                        break
                results.append(self.tok_decode(inp[len(enc) :]))
            return results

    wrapped = SlotMambaLM(model, config)
    tasks = config.eval_harness_tasks.split(",")

    task_results = lm_eval.simple_evaluate(
        model=wrapped,
        tasks=tasks,
        batch_size=config.eval_batch_size,
    )

    per_task = {}
    for task_name, task_data in task_results["results"].items():
        per_task[task_name] = task_data["acc_norm,none"]

    return {
        "per_task": per_task,
        "avg_accuracy": sum(per_task.values()) / len(per_task),
    }


def run_evaluation(model: nn.Module, config: Config) -> dict:
    model.eval()
    if config.task == "lm":
        if config.eval_mode == "lm_harness":
            result = evaluate_harness(model, config)
            result["mode"] = "lm_harness"
        else:
            result = evaluate_perplexity(model, config)
            result["mode"] = "lm_perplexity"
    elif config.eval_mode == "capacity_cliff":
        result = capacity_cliff(model, config)
        result["mode"] = "capacity_cliff"
    elif config.eval_mode == "grid":
        result = multi_config_grid(model, config)
        result["mode"] = "grid"
    else:
        result = _aggregate_seeds(
            model,
            num_pairs=config.num_pairs,
            vocab_size=config.vocab_size,
            base_seed=config.seed,
            num_samples=config.eval_num_samples,
            batch_size=config.eval_batch_size,
            num_eval_seeds=config.num_eval_seeds,
        )
        result["mode"] = "standard"

    result["d_state"] = config.d_state
    result["d_model"] = config.d_model
    result["encoder_layers"] = config.encoder_layers
    result["decoder_layers"] = config.decoder_layers
    return result
