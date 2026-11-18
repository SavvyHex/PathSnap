"""Write the model experiment tables for the team report from measured results."""
import csv
from pathlib import Path
from src.config import RESULTS


def rows(filename):
    with (RESULTS / filename).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def main():
    baseline = rows("model_comparison.csv")
    variants = rows("tuned_vs_untuned.csv")
    columns = [("model","Model"),("cv_macro_f1_mean","CV Macro F1"),
               ("train_macro_f1","Training Macro F1"),("macro_f1","Student-test Macro F1"),
               ("top1_accuracy","Top-1"),("log_loss","Log loss"),
               ("route_continuity","Continuity"),("inference_s","Inference seconds")]
    lines = ["# PathSnap model experiments", "",
             "Five-fold validation keeps complete trips together. The comparison uses every tenth training observation and evaluates all ten student-test trips. The nearest candidate appearing in the reference route supplies the pointwise label. Scores measure agreement with that proxy.", "",
             "| " + " | ".join(v for _,v in columns) + " |",
             "|"+"---|"*len(columns)]
    for row in baseline:
        cells=[row["model"]]+[f"{float(row[k]):.4f}" for k,_ in columns[1:]]
        lines.append("| "+" | ".join(cells)+" |")
    lines.extend(["","## Bayesian tuning","",
                  "| Variant | Model | CV Macro F1 | Student-test Top-1 | Student-test Macro F1 | Log loss |",
                  "|---|---|---:|---:|---:|---:|"])
    for v in variants:
        lines.append(f"| {v['variant']} | {v['model']} | {float(v['cv_macro_f1']):.4f} | {float(v['top1_accuracy']):.4f} | {float(v['macro_f1']):.4f} | {float(v['log_loss']):.4f} |")
    lines.extend(["",f"Student-test GPS observations: {variants[-1]['n_observations']}. Observations without candidates: {variants[-1]['empty_candidate_sets']}.",
                  "", "Full fold scores, continuity, timing and model configurations are in model_comparison.csv. Trial parameters and objectives are in optuna_trials.csv."])
    (RESULTS/"member2_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
