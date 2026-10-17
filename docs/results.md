# Results tables

`lilpipe results` collects numeric metrics from lm-eval-style JSON artifacts and
renders a compact Markdown, CSV, or JSON table.

```sh
lilpipe results configs/results/main.yml
lilpipe results configs/results/main.yml --format csv
lilpipe results configs/results/main.yml --format json
```

As with pipelines, relative paths resolve from the current working directory.

## Configuration

A version-1 results config defines display columns, experiment rows, and the
artifact location for each metric:

```yaml
version: 1
id: minimal-results

columns:
  - id: accuracy
    label: Accuracy
    direction: maximize
    format: percent
    precision: 1

rows:
  - id: base
    label: Toy base
    root: artifacts/evals/toy-base
    group: baselines
  - id: tuned
    label: Toy tuned
    root: artifacts/evals/toy-tuned
    group: derived

metrics:
  - column: accuracy
    task: toy_task
    key: "acc,none"
    files: ["toy-score/results*.json"]
```

Each column has a unique ID, label, optimization direction (`maximize` or
`minimize`), display format (`percent` or `number`), and non-negative decimal
precision. Percent values are multiplied by 100 for Markdown and CSV display;
JSON retains the original number.

Each row defines its label and artifact root. `group` is optional. Markdown
inserts a blank separator whenever adjacent rows change group; grouping does not
change CSV or JSON.

## Artifact discovery

Every metric maps one column to an lm-eval task and key. Its `files` entries are
glob patterns relative to each row's root. Patterns must be relative and cannot
contain `..`.

For every row and metric, lilpipe:

1. expands all configured patterns and deduplicates matching files;
2. ignores files that do not contain the configured task and key; and
3. selects the newest matching file that contains a numeric value.

A compatible artifact looks like:

```json
{
  "results": {
    "toy_task": {
      "acc,none": 0.75
    }
  }
}
```

Malformed JSON and non-numeric configured values are errors with row and column
context. A missing artifact, task, or key is not an error: Markdown displays an
em dash, CSV uses an empty cell, and JSON uses `null`.

## Best values

Markdown bolds the best available value in each column according to its
direction. All ties are highlighted. Missing values do not participate. CSV and
JSON contain plain values without highlighting.
