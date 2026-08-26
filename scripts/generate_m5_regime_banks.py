#!/usr/bin/env python3
"""
Deterministic regime-specific scenario bank generator for M5.

The M5 formal experiment compares four cost regimes on the SAME physical
scenario trajectories.  The production environment (src/envs/selective_
maintenance_env.py:_validate_scenario) rejects a scenario when its
``cost_regime_id`` field disagrees with the effective regime selected on
the command line.  The physical base banks
(``configs/scenarios/m5_pilot_k{1,2}.json`` and ``m5_validation_k{1,2}.json``)
embed ``cost_regime_id="failure-light-no-waste"`` and therefore only allow
the failure-light-no-waste regime.

This generator produces regime-specific derived banks where the ONLY
fields that change relative to the physical base bank are:

  - ``bank_id``                  (must encode the regime)
  - each ``scenario.scenario_id`` (must encode the regime to keep IDs unique)
  - each ``scenario.cost_regime_id`` (the whole point of the derivation)

Every other physical field (initial_unit_ids, initial_cycles,
replacement_seed, environment_seed, episode_horizon,
maintenance_capacity, split) is preserved exactly so that for a fixed
(K, seed) the four regime banks drive the same physical episode
trajectories, predictor/cache lookups, failure processes and action table.

The generator:
  - reads the physical base bank;
  - preserves all physical scenario content exactly;
  - updates only the permitted regime/provenance fields;
  - writes canonical deterministic JSON (indent=2, sort_keys=True, trailing
    newline) so that file bytes are stable across machines;
  - is idempotent (re-running produces byte-identical output);
  - fails if an unexpected field would change;
  - prints complete SHA256 hashes;
  - supports a verify-only mode that confirms checked-in assets match
    generation.

Run modes:

  python scripts/generate_m5_regime_banks.py           # generate + verify
  python scripts/generate_m5_regime_banks.py --verify # verify-only (no write)
  python scripts/generate_m5_regime_banks.py --help

Environment validation contract that this generator respects:

  src/envs/selective_maintenance_env.py:698-705 -- cost_regime_id match
  src/envs/scenario_bank.py:42-149             -- Scenario fields + serialization
  src/envs/costs.py:86-112                     -- valid regime registry
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Add repository root to path for src. imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.scenario_bank import ScenarioBank
from src.envs.costs import COST_REGIMES


# ---------------------------------------------------------------------------
# Frozen definition of the regime-bank asset contract.
# ---------------------------------------------------------------------------

# Mapping: physical base bank -> (K, split).
BASE_BANKS: list[tuple[str, int, str]] = [
    # (base path, K, split)
    ("configs/scenarios/m5_pilot_k1.json", 1, "predictor_train"),
    ("configs/scenarios/m5_pilot_k2.json", 2, "predictor_train"),
    ("configs/scenarios/m5_validation_k1.json", 1, "rl_validation"),
    ("configs/scenarios/m5_validation_k2.json", 2, "rl_validation"),
]

# The physical fields that MUST be byte-identical across all four regimes
# for a given (base bank).  Any divergence here would alter physical
# trajectories and violates the frozen scientific design.
PHYSICAL_SCENARIO_FIELDS = frozenset({
    "initial_unit_ids",
    "initial_cycles",
    "replacement_seed",
    "environment_seed",
    "episode_horizon",
    "maintenance_capacity",
    "split",
})

# The only scenario fields permitted to change across regime variants.
PERMITTED_CHANGING_FIELDS = frozenset({
    "scenario_id",
    "cost_regime_id",
})


# ---------------------------------------------------------------------------
# Naming helpers.
# ---------------------------------------------------------------------------

def regime_short(regime_id: str) -> str:
    """Map a cost regime id to a short stable suffix used in bank/scenario ids.

    The suffixes are deterministic string transforms that preserve the regime
    identity and remain human-readable.  Example:
        failure-light-no-waste      -> light
        failure-heavy-no-waste       -> heavy
        failure-light-waste-aware    -> light_waste
        failure-heavy-waste-aware    -> heavy_waste
    """
    suffix_map = {
        "failure-light-no-waste": "light",
        "failure-heavy-no-waste": "heavy",
        "failure-light-waste-aware": "light_waste",
        "failure-heavy-waste-aware": "heavy_waste",
    }
    if regime_id not in suffix_map:
        raise ValueError(f"Unknown cost regime for suffix mapping: {regime_id}")
    return suffix_map[regime_id]


def regime_bank_id(base_bank_id: str, regime_id: str) -> str:
    """Derive a regime-specific bank id from the physical base bank id."""
    return f"{base_bank_id}__{regime_short(regime_id)}"


def regime_scenario_id(base_scenario_id: str, regime_id: str) -> str:
    """Derive a regime-specific scenario id from the physical base id."""
    return f"{base_scenario_id}__{regime_short(regime_id)}"


def regime_bank_path(base_bank_path: str, regime_id: str) -> str:
    """Derive the regime-specific bank file path from the base path.

    The convention is:
        configs/scenarios/m5_pilot_k1.json ->
        configs/scenarios/m5_pilot_k1__light.json
        configs/scenarios/m5_pilot_k1.json ->
        configs/scenarios/m5_pilot_k1__heavy.json
        configs/scenarios/m5_pilot_k1.json ->
        configs/scenarios/m5_pilot_k1__light_waste.json
        configs/scenarios/m5_pilot_k1.json ->
        configs/scenarios/m5_pilot_k1__heavy_waste.json
    """
    p = Path(base_bank_path)
    return str(p.parent / f"{p.stem}__{regime_short(regime_id)}{p.suffix}")


# ---------------------------------------------------------------------------
# Canonical deterministic JSON serialization.
# ---------------------------------------------------------------------------

def canonical_dump(obj) -> str:
    """Canonical deterministic JSON: indent=2, sort_keys=True, no trailing NL."""
    return json.dumps(obj, indent=2, sort_keys=True, separators=(",", ": "))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------
# Core derivation + verification.
# ---------------------------------------------------------------------------

def derive_regime_bank(base_bank_path: str, regime_id: str) -> tuple[dict, dict]:
    """Read the base bank and produce the regime-specific bank dict.

    Returns:
        (derived_dict, physical_extraction)
        derived_dict is the canonical dataclass-free dict to be written as JSON.
        physical_extraction records the physical fields for cross-regime comparison.
    """
    base = json.loads(Path(base_bank_path).read_text(encoding="utf-8"))

    # Confirm regime id is known so we fail closed on typo.
    if regime_id not in COST_REGIMES:
        raise ValueError(
            f"Unknown cost regime '{regime_id}'. Available: {sorted(COST_REGIMES.keys())}"
        )

    # Derive regime-scoped bank id + scenario ids.
    new_bank_id = regime_bank_id(base["bank_id"], regime_id)

    derived_scenarios: list[dict] = []
    physical_extraction: dict[str, dict] = {}

    for s in base["scenarios"]:
        # Defensive: every scenario must declare cost_regime_id.
        if "cost_regime_id" not in s:
            raise ValueError(
                f"Base scenario {s.get('scenario_id')!r} missing cost_regime_id"
            )

        # Extract the physical fields we must preserve exactly.
        physical = {k: s[k] for k in sorted(PHYSICAL_SCENARIO_FIELDS)}
        # Record under the BASE scenario id so cross-regime comparison is stable
        # regardless of the regime suffix appended to scenario_id in variants.
        physical_extraction[s["scenario_id"]] = physical

        # Build the derived scenario: copy base, override permitted fields, fail
        # closed if an unexpected key would be touched.
        new_scenario: dict = dict(s)
        new_scenario["cost_regime_id"] = regime_id
        new_scenario["scenario_id"] = regime_scenario_id(s["scenario_id"], regime_id)

        # Sanity: a scenario may also carry extra unknown keys.  Those are
        # carried through unchanged.  The contract is that ONLY the two
        # permitted changing fields differ between regimes, which the verify
        # step below enforces.
        derived_scenarios.append(new_scenario)

    derived = {
        "bank_id": new_bank_id,
        "split": base["split"],
        "scenarios": derived_scenarios,
    }

    return derived, physical_extraction


def write_canonical(derived: dict, path: Path) -> bytes:
    """Write canonical JSON and return the bytes written (incl. trailing newline)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical_dump(derived) + "\n"
    data = text.encode("utf-8")
    path.write_bytes(data)
    return data


def verify_bank_on_disk(path: Path, derived: dict) -> tuple[bool, str]:
    """Compare a bank on disk to the derived dict.

    Returns (ok, message).  ok is True only when the disk bytes equal the
    canonical bytes derived in-memory.
    """
    if not path.exists():
        return False, f"{path}: file does not exist"
    disk_bytes = path.read_bytes()
    expected = (canonical_dump(derived) + "\n").encode("utf-8")
    disk_sha = sha256_bytes(disk_bytes)
    expected_sha = sha256_bytes(expected)
    if disk_sha == expected_sha:
        return True, f"{path}: OK sha={disk_sha}"
    return False, (
        f"{path}: MISMATCH disk={disk_sha} expected={expected_sha}"
    )


# ---------------------------------------------------------------------------
# Cross-regime physical-equivalence comparison (used by tests + verify).
# ---------------------------------------------------------------------------

def extract_physicals(bank_dict: dict) -> dict[str, dict]:
    """Extract the physical fields for each scenario of a bank dict."""
    out: dict[str, dict] = {}
    for s in bank_dict["scenarios"]:
        # Use the BASE id (strip regime suffix) so cross-regime comparison
        # is keyed by the same physical identity.
        base_id = s["scenario_id"].split("__")[0]
        out[base_id] = {k: s[k] for k in sorted(PHYSICAL_SCENARIO_FIELDS)}
    return out


def assert_physicals_identical(regime_dicts: list[tuple[str, dict]]) -> None:
    """Fail if any two regime-derived banks differ on physical fields.

    regime_dicts is a list of (regime_id, derived_dict).  The physical
    extractions keyed by base scenario id must all be byte-identical.
    """
    if len(regime_dicts) < 2:
        return
    base = extract_physicals(regime_dicts[0][1])
    for regime_id, d in regime_dicts[1:]:
        cur = extract_physicals(d)
        if cur != base:
            # Identify which scenario differs for a clear error.
            differing = []
            for sid in sorted(set(base.keys()) | set(cur.keys())):
                if base.get(sid) != cur.get(sid):
                    differing.append(
                        f"scenario {sid}: base={base.get(sid)} vs {regime_id}={cur.get(sid)}"
                    )
            raise AssertionError(
                "Physical fields are NOT identical across regimes:\n  - "
                + "\n  - ".join(differing)
            )


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def generate_all(verify_only: bool = False) -> dict:
    """Generate or verify the entire regime-bank asset set.

    Returns a report dict with hashes and a per-asset status list.
    """
    report: dict = {"assets": [], "physical_identity_ok": False}

    # For each base bank, derive the four regime variants and (optionally)
    # write canonical JSON.  Cross-regime physical equality is asserted for
    # each base bank independently.
    for base_path, _, _ in BASE_BANKS:
        regime_dicts: list[tuple[str, dict]] = []
        for regime_id in sorted(COST_REGIMES):
            derived, _ = derive_regime_bank(base_path, regime_id)
            regime_dicts.append((regime_id, derived))

        assert_physicals_identical(regime_dicts)

        for regime_id, derived in regime_dicts:
            out_path = Path(regime_bank_path(base_path, regime_id))
            if not verify_only:
                bytes_written = write_canonical(derived, out_path)
                sha = sha256_bytes(bytes_written)
                report["assets"].append({
                    "base_bank_path": base_path,
                    "regime_id": regime_id,
                    "derived_bank_path": str(out_path),
                    "bank_id": derived["bank_id"],
                    "sha256": sha,
                    "wrote": True,
                })
            else:
                ok, msg = verify_bank_on_disk(out_path, derived)
                sha = msg.split("sha=")[-1] if "sha=" in msg else None
                report["assets"].append({
                    "base_bank_path": base_path,
                    "regime_id": regime_id,
                    "derived_bank_path": str(out_path),
                    "bank_id": derived["bank_id"],
                    "sha256": sha,
                    "wrote": False,
                    "verify_ok": ok,
                    "verify_msg": msg,
                })

    report["physical_identity_ok"] = True
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic regime-specific scenario bank generator (M5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify only: confirm checked-in regime banks match in-memory generation, do not write.",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print the full report with SHA256 hashes.",
    )
    args = parser.parse_args()

    report = generate_all(verify_only=args.verify)

    fails: list[str] = []
    print("=" * 70)
    print("M5 regime-specific scenario bank generator")
    print("=" * 70)
    if args.verify:
        print("MODE: VERIFY ONLY (no writes)")
    else:
        print("MODE: GENERATE + VERIFY")
    print()

    for a in report["assets"]:
        sha = a.get("sha256") or ""
        if args.verify:
            ok = bool(a.get("verify_ok"))
            tag = "OK   " if ok else "FAIL "
            line = f"{tag} {a['derived_bank_path']}  {sha}"
            print(line)
            if not ok:
                fails.append(a.get("verify_msg", a['derived_bank_path']))
        else:
            line = f"WROTE {a['derived_bank_path']}  {sha}"
            print(line)
        print(f"      base={a['base_bank_path']}  regime={a['regime_id']}  bank_id={a['bank_id']}")

    print()
    print(f"Physical field identity across regimes: "
          f"{'OK' if report['physical_identity_ok'] else 'FAIL'}")

    # If we wrote, also run an immediate verify to confirm idempotency on disk.
    if not args.verify:
        vreport = generate_all(verify_only=True)
        for a in vreport["assets"]:
            if not a.get("verify_ok"):
                fails.append(a.get("verify_msg", a['derived_bank_path']))

    if fails:
        print()
        print("VERIFY FAILURES:")
        for f in fails:
            print(f"  - {f}")
        return 1

    print()
    print("All regime-specific banks verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
