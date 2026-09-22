from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .io import read_json
from .paths import dataset_paths


Record = dict[str, Any]


class Dataset:
    def __init__(self, config: dict[str, Any]):
        paths = dataset_paths(config)
        self.paths = paths
        self.questions: list[Record] = read_json(paths["questions"])
        self.gold: list[Record] = read_json(paths["gold"])
        if not isinstance(self.questions, list) or not isinstance(self.gold, list):
            raise ValueError("V5.2 questions and gold must both be JSON arrays")
        self.question_by_id = self._index(self.questions, "public question")
        self.gold_by_id = self._index(self.gold, "private gold")
        self.validate_core_alignment()

    @staticmethod
    def _index(rows: list[Record], label: str) -> dict[str, Record]:
        out: dict[str, Record] = {}
        for row in rows:
            if not isinstance(row, dict) or not row.get("id"):
                raise ValueError(f"Invalid {label} record without id")
            item_id = str(row["id"])
            if item_id in out:
                raise ValueError(f"Duplicate {label} id: {item_id}")
            out[item_id] = row
        return out

    def validate_core_alignment(self) -> None:
        question_ids = set(self.question_by_id)
        gold_ids = set(self.gold_by_id)
        if question_ids != gold_ids:
            missing_gold = sorted(question_ids - gold_ids)[:10]
            missing_questions = sorted(gold_ids - question_ids)[:10]
            raise ValueError(
                "Public/private ID mismatch; "
                f"missing_gold={missing_gold}, missing_questions={missing_questions}"
            )
        expected_core_size = int(self.paths.get("expected_core_size", 6200))
        if len(self.questions) != expected_core_size or len(self.gold) != expected_core_size:
            raise ValueError(
                f"Expected core size {expected_core_size}, got questions={len(self.questions)}, gold={len(self.gold)}"
            )
        for row in self.questions:
            if row.get("benchmark_version") != "5.2":
                raise ValueError(f"Unexpected benchmark version for {row['id']}")
            forbidden = {"answer", "scoring", "explanation", "evidence", "verification", "metadata"}
            leaked = forbidden.intersection(row)
            if leaked:
                raise ValueError(f"Public question {row['id']} contains forbidden keys: {sorted(leaked)}")
        for row in self.gold:
            if str(row.get("id")) not in self.question_by_id:
                raise ValueError(f"Private gold {row.get('id')} has no public question")
            public = self.question_by_id[str(row["id"])]
            public_group = (public.get("group_ids") or {}).get("knowledge_unit_id")
            gold_group = row.get("knowledge_unit_id")
            if public_group and gold_group and str(public_group) != str(gold_group):
                raise ValueError(f"Knowledge-unit mismatch for {row['id']}: public={public_group} gold={gold_group}")

    def profile(self) -> dict[str, Any]:
        return {
            "questions": len(self.questions),
            "gold": len(self.gold),
            "task_format_counts": dict(sorted(Counter(str(x.get("task_format")) for x in self.questions).items())),
            "type_counts": dict(sorted(Counter(str(x.get("type")) for x in self.questions).items())),
            "knowledge_units": len({x["group_ids"]["knowledge_unit_id"] for x in self.questions}),
            "diagnostic_counts": self.diagnostic_counts(),
        }

    def load_diagnostic(self, name: str, private: bool = False) -> list[Record]:
        subdir = self.paths["diagnostic_private"] if private else self.paths["diagnostic_public"]
        filename = {
            "compositional": "compositional_bundles_gold.json" if private else "compositional_bundles.json",
            "counterfactual": "counterfactual_triplets_gold.json" if private else "counterfactual_triplets.json",
            "answer_format": "answer_format_variants_gold.json" if private else "answer_format_variants.json",
        }.get(name)
        if not filename:
            raise ValueError(f"Unknown diagnostic set: {name}")
        rows = read_json(Path(subdir) / filename)
        if not isinstance(rows, list):
            raise ValueError(f"Diagnostic set must be a JSON array: {filename}")
        return rows

    def diagnostic_counts(self) -> dict[str, int]:
        return {
            "compositional_bundles": len(self.load_diagnostic("compositional", private=False)),
            "counterfactual_triplets": len(self.load_diagnostic("counterfactual", private=False)),
            "answer_format_semantic_groups": len(self.load_diagnostic("answer_format", private=False)),
        }

    def validate_diagnostics(self) -> dict[str, Any]:
        compositional = self.load_diagnostic("compositional", private=False)
        compositional_gold = {str(row["id"]): row for row in self.load_diagnostic("compositional", private=True)}
        counterfactual = self.load_diagnostic("counterfactual", private=False)
        counterfactual_gold = {str(row["id"]): row for row in self.load_diagnostic("counterfactual", private=True)}
        answer_format = self.load_diagnostic("answer_format", private=False)
        answer_format_gold = {str(row["id"]): row for row in self.load_diagnostic("answer_format", private=True)}
        errors: list[str] = []
        if len(compositional) != 451:
            errors.append(f"Expected 451 compositional bundles, got {len(compositional)}")
        if len(counterfactual) != 451:
            errors.append(f"Expected 451 counterfactual triplets, got {len(counterfactual)}")
        if len(answer_format) != 2877:
            errors.append(f"Expected 2877 answer-format groups, got {len(answer_format)}")
        for row in compositional:
            bundle_id = str(row.get("bundle_id", ""))
            if bundle_id not in compositional_gold:
                errors.append(f"Missing compositional gold for {bundle_id}")
                continue
            if str(row.get("target_item_id", "")) not in self.question_by_id:
                errors.append(f"Compositional bundle {bundle_id} has unknown target_item_id")
            atomic_ids = {str(x.get("atomic_id", "")) for x in row.get("atomic_questions", [])}
            gold_atomic_ids = {str(x.get("atomic_id", "")) for x in compositional_gold[bundle_id].get("atomic_answers", [])}
            if atomic_ids != gold_atomic_ids:
                errors.append(f"Compositional bundle {bundle_id} atomic ID mismatch")
        for row in counterfactual:
            triplet_id = str(row.get("triplet_id", ""))
            if triplet_id not in counterfactual_gold:
                errors.append(f"Missing counterfactual gold for {triplet_id}")
                continue
            for variant in ("original", "lexical_control", "causal_intervention"):
                if not row.get(variant, {}).get("question"):
                    errors.append(f"Counterfactual triplet {triplet_id} missing {variant} question")
                if not counterfactual_gold[triplet_id].get(variant, {}).get("accepted_answers"):
                    errors.append(f"Counterfactual triplet {triplet_id} missing {variant} gold")
        for row in answer_format:
            group_id = str(row.get("format_group_id", ""))
            gold = answer_format_gold.get(group_id)
            if gold is None:
                errors.append(f"Missing answer-format gold for {group_id}")
                continue
            if not row.get("open_ended", {}).get("question"):
                errors.append(f"Answer-format group {group_id} missing open-ended question")
            correct_labels = gold.get("correct_labels") or {}
            for permutation in row.get("mcq_permutations", []):
                permutation_id = str(permutation.get("permutation_id", ""))
                labels = {str(option.get("label", "")) for option in permutation.get("options", [])}
                correct_label = str(correct_labels.get(permutation_id, ""))
                if correct_label not in labels:
                    errors.append(f"Answer-format group {group_id} invalid correct label for {permutation_id}")
                correct_option_id = str(gold.get("correct_option_id", ""))
                option_ids = {str(option.get("option_id", "")) for option in permutation.get("options", [])}
                if correct_option_id not in option_ids:
                    errors.append(f"Answer-format group {group_id} missing correct option_id in {permutation_id}")
        if errors:
            raise ValueError("Diagnostic validation failed: " + "; ".join(errors[:20]))
        return {
            "compositional_bundles": len(compositional),
            "counterfactual_triplets": len(counterfactual),
            "answer_format_semantic_groups": len(answer_format),
            "status": "ok",
        }
