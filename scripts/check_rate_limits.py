#!/usr/bin/env python3
"""
Check current rate limit status
Run this anytime to see how much quota you've used
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.rate_limiter import RateLimiter
from src.config import config, api_config


def _safe_state_path(model_name: str) -> Path:
    """Map model name to monitor state path."""
    safe_name = model_name.replace("-", "_").replace(".", "_")
    return Path(f"results/.monitor_state_{safe_name}.json")


def _load_monitor_state(model_name: str) -> dict:
    """Load persisted monitor state for a model if it exists."""
    path = _safe_state_path(model_name)
    if not path.exists():
        return {}
    try:
        with path.open() as handle:
            state = json.load(handle)
            state["_state_path"] = str(path)
            return state
    except Exception:
        return {}


def _print_monitor_summary(label: str, state: dict, fallback_limits: dict):
    """Print a concise daily usage summary from monitor state."""
    if not state:
        print(f"{label}: no monitor state found.")
        limits = fallback_limits
    else:
        limits = state.get("limits", {}) or fallback_limits
    
    current = state.get("current_day", {})
    rpm = limits.get("rpm")
    tpm = limits.get("tpm")
    rpd = limits.get("rpd")
    req_today = current.get("requests", 0)
    tokens_today = current.get("tokens", 0)
    remaining_rpd = rpd - req_today if rpd is not None else None
    
    print(f"{label}:")
    print(f"  Limits: RPM={rpm}, TPM={tpm}, RPD={rpd}")
    print(f"  Today: {req_today} requests, {tokens_today:,} tokens")
    if remaining_rpd is not None:
        print(f"  Remaining RPD today: {remaining_rpd}")
    if "_state_path" in state:
        print(f"  Source: {state['_state_path']}")
    
    return {
        "limits": limits,
        "requests_today": req_today,
        "tokens_today": tokens_today,
        "remaining_rpd": remaining_rpd,
        "state_path": state.get("_state_path"),
    }

def main():
    # Generation model (reads monitor state for real RPD)
    gen_model = config.model_name
    gen_state = _load_monitor_state(gen_model)
    limiter = RateLimiter(gen_model)
    gen_limits = limiter.limits.__dict__
    gen_summary = _print_monitor_summary("Generation model", gen_state, gen_limits)

    # Embedding model
    embed_model = config.embedding_model_name
    embed_state = _load_monitor_state(embed_model)
    embed_limits = {"rpm": 1500, "tpm": 10_000_000, "rpd": api_config.rate_limit_rpd}
    embed_summary = _print_monitor_summary("Embedding model", embed_state, embed_limits)

    # Minute-level headroom snapshot using RateLimiter
    status = limiter.get_status()
    print("\nMinute-level headroom (limiter snapshot):")
    print(f"  Requests this minute: {status['current_usage']['requests_this_minute']}/{status['limits']['rpm']}")
    print(f"  Tokens this minute: {status['current_usage']['tokens_this_minute']:,}/{status['limits']['tpm']:,}")

    warnings = []
    if gen_summary and gen_summary["remaining_rpd"] is not None and gen_summary["limits"].get("rpd"):
        used_pct = (gen_summary["requests_today"] / gen_summary["limits"]["rpd"]) * 100
        if used_pct > 95:
            warnings.append("  • ❌ Generation RPD >95%: stop for today.")
        elif used_pct > 80:
            warnings.append(f"  • Generation RPD at {used_pct:.0f}%: slow down.")
    if embed_summary and embed_summary["remaining_rpd"] is not None and embed_summary["limits"].get("rpd"):
        used_pct = (embed_summary["requests_today"] / embed_summary["limits"]["rpd"]) * 100
        if used_pct > 95:
            warnings.append("  • ❌ Embedding RPD >95%: stop embedding calls today.")
        elif used_pct > 80:
            warnings.append(f"  • Embedding RPD at {used_pct:.0f}%: limit embedding calls.")

    print("\n⚠️  WARNINGS:")
    if warnings:
        print("\n".join(warnings))
    else:
        print("  ✓ All limits healthy\n")

    # Persist combined snapshot
    status_file = Path("results/.rate_limit_status.json")
    status_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generation": gen_summary,
        "embedding": embed_summary,
        "limiter_snapshot": status,
    }
    with status_file.open("w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Status saved to: {status_file}")

if __name__ == "__main__":
    main()
