#!/usr/bin/env python3
"""Build reproducible 50-disease TQ sets from MediQ MCQ option labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


DIRECT_DIAGNOSIS_PATTERNS = (
    "most likely diagnosis",
    "most likely underlying diagnosis",
    "patient's most likely diagnosis",
    "most likely condition?",
    "most likely underlying condition",
    "which personality disorder",
    "personality disorders best characterizes",
    "type of shock best describes",
    "pathologies most likely underlies",
    "medical conditions is the physician referring to",
    "best describes the most likely diagnosis",
    "most likely the diagnosis",
    "most probable diagnosis",
)

DERMATOLOGY_IMEDQA_DISEASE_OPTIONS = (
    "Actinic keratosis",
    "Alopecia areata",
    "Basal cell carcinoma",
    "Chromoblastomycosis",
    "Differentiated squamous cell carcinoma (SCC)",
    "Malignant melanoma",
    "Rubella",
    "Squamous cell carcinoma",
    "Tinea capitis",
    "Vitiligo",
)

NEUROLOGY_DISEASE_OPTIONS = (
    "Acoustic neuroma",
    "Acute labyrinthitis",
    "Alcohol withdrawal",
    "Alcohol-related dementia",
    "Alzheimer's disease",
    "Amyotrophic lateral sclerosis",
    "Arnold-Chiari malformation type 1",
    "Arnold-Chiari malformation type 2",
    "Arnold-Chiari malformation type 3",
    "Arnold-Chiari malformation type 4",
    "Ataxia-telangiectasia",
    "Benign paroxysmal positional vertigo",
    "Brain metastasis",
    "Brain tumor",
    "Cerebellar tumor",
    "Cerebral salt wasting syndrome",
    "Cerebral venous thrombosis",
    "Cluster headache",
    "Conduction aphasia",
    "Craniopharyngioma",
    "Dandy-Walker malformation",
    "Delirium",
    "Delirium tremens",
    "Demyelinating disease",
    "Epidural hematoma",
    "Glioblastoma multiforme",
    "Global aphasia",
    "Huntington disease",
    "Hydrocephalus",
    "Idiopathic intracranial hypertension",
    "Ischemic stroke",
    "Lewy body dementia",
    "Meniere disease",
    "Meningioma",
    "Meningitis",
    "Migraine",
    "Multiple sclerosis",
    "Neonatal encephalopathy",
    "Neurofibromatosis type 1",
    "Neurofibromatosis Type II",
    "Neuroleptic malignant syndrome",
    "Normal pressure hydrocephalus",
    "Oligodendroglioma",
    "Osmotic demyelination syndrome",
    "Pineal tumor",
    "Pituitary adenoma",
    "Postherpetic neuralgia",
    "Primary seizure disorder",
    "Rett syndrome",
    "Schwannoma",
    "Spinal cord compression",
    "Spinal epidural hematoma",
    "Sturge-Weber syndrome",
    "Subarachnoid hemorrhage",
    "Subdural hematoma",
    "Transcortical sensory aphasia",
    "Transient ischemic attack",
    "Trigeminal neuralgia",
    "Tuberous sclerosis complex",
    "Vascular dementia",
    "Viral meningitis",
    "Wernicke aphasia",
    "Wernicke's encephalopathy",
)

GASTROENTEROLOGY_DISEASE_OPTIONS = (
    "Acute cholangitis",
    "Acute diverticulitis",
    "Acute mesenteric ischemia",
    "Acute pancreatitis",
    "Amebiasis",
    "Aphthous stomatitis",
    "Appendicitis",
    "Bacterial overgrowth",
    "Barrett’s esophagus",
    "Bowel adhesions",
    "Celiac disease",
    "Cholangiocarcinoma",
    "Chronic hepatitis C infection",
    "Chronic pancreatitis",
    "Chronic pancreatitis with pseudocyst rupture",
    "Colon cancer",
    "Crohn's disease",
    "Cyclic vomiting syndrome",
    "Diverticulosis",
    "Dubin-Johnson syndrome",
    "Emphysematous cholecystitis",
    "Gallstone ileus",
    "Gastric adenocarcinoma",
    "Gastric cancer",
    "Gastrin-secreting tumor",
    "Gastroenteritis",
    "Gastroesophageal reflux disease",
    "Giardiasis",
    "Gilbert syndrome",
    "H. pylori infection",
    "Hemochromatosis",
    "Hepatic adenoma",
    "Hepatic encephalopathy",
    "Hepatitis A",
    "Hepatitis B",
    "Hepatitis C",
    "Hepatitis D",
    "Hepatitis E",
    "Hepatoblastoma",
    "Hepatocellular carcinoma",
    "Hepatorenal syndrome",
    "Hypertrophic pyloric stenosis",
    "Infectious colitis",
    "Intestinal ischemia",
    "Intestinal type gastric adenocarcinoma",
    "Irritable bowel syndrome (IBS)",
    "Liver failure",
    "Malignant biliary tract obstruction",
    "Multiple hepatic cysts",
    "Necrotizing enterocolitis",
    "NSAID induced peptic ulcer disease",
    "Peptic ulcer disease",
    "Portal hypertension",
    "Primary sclerosing cholangitis",
    "Pseudomembranous colitis and diarrhea",
    "Ruptured appendix",
    "Salmonellosis",
    "Spontaneous bacterial peritonitis",
    'Squamous cell carcinoma of the esophagus "',
    "Traveler’s diarrhea due to ETEC",
    "Traveler’s diarrhea due to Norovirus",
    "Volvulus",
    "Zollinger-Ellison syndrome",
)

DOMAIN_SPECS = {
    "dermatology": {"specialty": "Dermatology", "seed_offset": 0},
    "neurology": {"specialty": "Neurology", "seed_offset": 1},
    "gastroenterology": {"specialty": "Gastroenterology", "seed_offset": 2},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imedqa", type=Path, required=True)
    parser.add_argument("--craft", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--set-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260923)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize_label(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip()
    text = text.replace("–", "-").replace("—", "-").replace("’", "'")
    return re.sub(r"\s+", " ", text).rstrip(". ").casefold()


def is_direct_diagnosis(question: str) -> bool:
    question = normalize_label(question)
    return any(pattern in question for pattern in DIRECT_DIAGNOSIS_PATTERNS)


def canonical_option_labels(samples: list[dict]) -> dict[str, str]:
    forms: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        for option in sample["options"].values():
            stripped = re.sub(r"\s+", " ", option.strip()).rstrip(". ")
            forms[normalize_label(stripped)][stripped] += 1
    return {
        normalized: sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        for normalized, counts in forms.items()
    }


def correct_label(sample: dict) -> str:
    return normalize_label(sample["options"][sample["answer_idx"]])


def resolve_curated_pool(labels: tuple[str, ...], canonical: dict[str, str]) -> list[str]:
    normalized = [normalize_label(label) for label in labels]
    missing = sorted(label for label in normalized if label not in canonical)
    if missing:
        raise ValueError(f"Curated labels absent from MediQ options: {missing}")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Curated pool contains duplicate normalized labels")
    return normalized


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    imedqa = load_jsonl(args.imedqa)
    craft = load_jsonl(args.craft)
    all_samples = imedqa + craft
    canonical = canonical_option_labels(all_samples)
    imedqa_diagnosis = [sample for sample in imedqa if is_direct_diagnosis(sample["question"])]

    craft_gt = {correct_label(sample) for sample in craft}
    dermatology_pool = sorted(
        craft_gt | set(resolve_curated_pool(DERMATOLOGY_IMEDQA_DISEASE_OPTIONS, canonical))
    )
    curated_pools = {
        "dermatology": dermatology_pool,
        "neurology": resolve_curated_pool(NEUROLOGY_DISEASE_OPTIONS, canonical),
        "gastroenterology": resolve_curated_pool(GASTROENTEROLOGY_DISEASE_OPTIONS, canonical),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created": "2026-09-23",
        "base_seed": args.seed,
        "set_size": args.set_size,
        "sampling": "uniform without replacement from manually audited disease-only MediQ MCQ option pools",
        "pool_policy": {
            "dermatology": "all iCRAFT-MD ground-truth diagnoses plus audited iMedQA dermatology disease options",
            "neurology": "audited neurology disease/condition labels appearing anywhere in MediQ MCQ options",
            "gastroenterology": "audited gastroenterology disease/condition labels appearing anywhere in MediQ MCQ options",
        },
        "domains": {},
    }

    for domain, spec in DOMAIN_SPECS.items():
        normalized_pool = sorted(curated_pools[domain])
        if len(normalized_pool) < args.set_size:
            raise ValueError(
                f"{domain} has {len(normalized_pool)} audited disease labels, "
                f"fewer than requested set size {args.set_size}"
            )

        domain_seed = args.seed + int(spec["seed_offset"])
        selected_normalized = random.Random(domain_seed).sample(
            normalized_pool, args.set_size
        )
        selected = sorted(canonical[label] for label in selected_normalized)
        output_path = args.output_dir / f"diseases_{domain}_50.json"
        output_path.write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n")

        eval_samples = [
            sample
            for sample in imedqa_diagnosis
            if spec["specialty"] in sample.get("patient", {}).get("specialties", [])
        ]
        if domain == "dermatology":
            eval_samples = craft + eval_samples
        gt_counts = Counter(correct_label(sample) for sample in eval_samples)
        selected_set = set(selected_normalized)
        covered_cases = sum(
            count for label, count in gt_counts.items() if label in selected_set
        )
        selected_gt = sorted(selected_set & set(gt_counts))
        manifest["domains"][domain] = {
            "seed": domain_seed,
            "audited_option_pool_size": len(normalized_pool),
            "evaluation_cases": len(eval_samples),
            "evaluation_ground_truth_label_count": len(gt_counts),
            "selected_ground_truth_label_count": len(selected_gt),
            "selected_ground_truth_labels": [canonical[label] for label in selected_gt],
            "covered_evaluation_cases": covered_cases,
            "uncovered_evaluation_cases": len(eval_samples) - covered_cases,
            "output_path": str(output_path),
            "sha256": sha256(output_path),
        }

    manifest_path = args.output_dir / "diseases_mediq_50_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
