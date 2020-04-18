# RAMP starting kit on the Fake News Detection

Fake news is intentionally written to mislead readers to believe false information, which makes it difficult and nontrivial to detect based on news content. Given the prevalence of this
new phenomenon, "Fake news" was even named the word of the year in 2016.

Authors: Emanuela Boros (LIMSI/CNRS), Balázs Kégl (LAL/CNRS), Roman Yurchak (Symerio)

## Getting started

### Install

To run a submission and the notebook you will need the dependencies listed
in `requirements.txt`. You can install install the dependencies with the
following command-line:

```bash
pip install -U -r requirements.txt
```

If you are using `conda`, we provide an `environment.yml` file for similar
usage.

### Challenge description

Get started on this RAMP with the
[dedicated notebook](fake_news_starting_kit.ipynb).

### Test a submission

The submissions need to be located in the `submissions` folder. For instance
for `my_submission`, it should be located in `submissions/my_submission`.

To run a specific submission, you can use the `ramp-test` command line:

```bash
ramp-test --submission my_submission
```

You can get more information regarding this command line:

```bash
ramp-test --help
```

### To go further

You can find more information regarding `ramp-workflow` and RAMP studio submissions in the
[dedicated documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)
