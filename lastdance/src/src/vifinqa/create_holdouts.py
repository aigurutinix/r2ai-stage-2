"""Create evaluation holdout sets from the base questions."""

import json
from pathlib import Path
import random

def create_holdouts(questions_path: Path, output_dir: Path) -> None:
    questions = []
    with open(questions_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
                
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Just a mock partitioning strategy for the capability evaluation
    random.shuffle(questions)
    
    template_holdout = questions[:50]
    entity_holdout = questions[50:100]
    temporal_holdout = questions[100:150]
    paraphrase_set = questions[150:200]
    synthetic_unseen = questions[200:250]
    
    def write_set(name: str, subset: list):
        with open(output_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for q in subset:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
                
    write_set("template_holdout", template_holdout)
    write_set("entity_holdout", entity_holdout)
    write_set("temporal_holdout", temporal_holdout)
    write_set("paraphrase_set", paraphrase_set)
    write_set("synthetic_unseen", synthetic_unseen)
    
    print(f"Created 5 holdout sets in {output_dir}")

if __name__ == "__main__":
    create_holdouts(Path("ViFinQA/questions/questions.jsonl"), Path("outputs/evaluation_sets"))
