import csv
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt

out_dir = Path("graphs")
out_dir.mkdir(exist_ok=True)

context_files = [
    "qwen3_context_overload_results_3trials.csv",
    "llama3_1_context_overload_results_3trials.csv",
]

position_files = [
    "qwen3_position_ablation_results_3trials.csv",
    "llama3_1_position_ablation_results_3trials.csv",
]


def read_csv_rows(filename):
    rows = []
    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


context_rows = []
for file in context_files:
    context_rows.extend(read_csv_rows(file))

position_rows = []
for file in position_files:
    position_rows.extend(read_csv_rows(file))


# Figure 1: Context-overload outcomes by model
context_counter = Counter(
    (row["Model"], row["Outcome Label"]) for row in context_rows
)

models = sorted(set(row["Model"] for row in context_rows))
outcomes = sorted(set(row["Outcome Label"] for row in context_rows))

x = range(len(models))
bar_width = 0.8 / max(1, len(outcomes))

fig, ax = plt.subplots(figsize=(8, 5))

for i, outcome in enumerate(outcomes):
    counts = [context_counter[(model, outcome)] for model in models]
    positions = [value + i * bar_width for value in x]
    ax.bar(positions, counts, width=bar_width, label=outcome)

ax.set_title("Context-Overload Outcomes by Model")
ax.set_xlabel("Model")
ax.set_ylabel("Number of outputs")
ax.set_xticks([value + bar_width * (len(outcomes) - 1) / 2 for value in x])
ax.set_xticklabels(models)
ax.legend(title="Outcome")
plt.tight_layout()
plt.savefig(out_dir / "figure1_context_overload_outcomes.png", dpi=300)
plt.close()



# Figure 2: Position-ablation outcomes
position_counter = Counter(
    (row["Model"], row["Injection Position"], row["Outcome Label"])
    for row in position_rows
)

labels = []
counts = []
outcome_labels = []

for row in position_rows:
    key = (row["Model"], row["Injection Position"], row["Outcome Label"])
    if key not in [(labels[i].split("\n")[0], labels[i].split("\n")[1], outcome_labels[i]) for i in range(len(labels))]:
        labels.append(f"{row['Model']}\n{row['Injection Position']}")
        counts.append(position_counter[key])
        outcome_labels.append(row["Outcome Label"])

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(labels, counts)

for i, outcome in enumerate(outcome_labels):
    ax.text(i, counts[i] + 0.05, outcome, ha="center", va="bottom", rotation=90, fontsize=8)

ax.set_title("Position Ablation: Outcome per Model and Position")
ax.set_xlabel("Model and injection position")
ax.set_ylabel("Number of outputs")
ax.set_ylim(0, max(counts) + 2)
plt.tight_layout()
plt.savefig(out_dir / "figure2_position_ablation_outcomes.png", dpi=300)
plt.close()


# Figure 3: Overall outcome distribution
all_rows = context_rows + position_rows
overall_counter = Counter(row["Outcome Label"] for row in all_rows)

overall_outcomes = list(overall_counter.keys())
overall_counts = [overall_counter[outcome] for outcome in overall_outcomes]

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(overall_outcomes, overall_counts)
ax.set_title("Outcome Distribution Across All Outputs")
ax.set_xlabel("Outcome label")
ax.set_ylabel("Number of outputs")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(out_dir / "figure3_all_outcomes_distribution.png", dpi=300)
plt.close()


# Figure 4: Harmful compliance vs all other outcomes
harmful_count = sum(
    1 for row in all_rows
    if row["Outcome Label"] == "POTENTIAL_HARMFUL_COMPLIANCE"
)

non_harmful_count = len(all_rows) - harmful_count

fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(
    ["Harmful compliance", "Noncompliance / failure mode"],
    [harmful_count, non_harmful_count]
)

ax.set_title("Overall Harmful Compliance")
ax.set_xlabel("Output category")
ax.set_ylabel("Number of outputs")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(out_dir / "figure4_harmful_compliance.png", dpi=300)
plt.close()

print("Graphs saved to the graphs folder.")