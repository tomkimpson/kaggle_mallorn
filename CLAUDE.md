Use 'bd' for task tracking. Be proactive - create beads when:
- Submitting SLURM jobs
- Starting multi-step experiments
- Leaving work incomplete
- Any task that spans sessions

## Kaggle Submission

To submit to the MALLORN competition:

```bash
source venv/bin/activate
export KAGGLE_API_TOKEN=KGAT_209f2e6f5fd3bc6c81588500c431b21c
kaggle competitions submit -c mallorn-astronomical-classification-challenge -f <submission_file.csv> -m "<message>"
```

Example:
```bash
source venv/bin/activate && export KAGGLE_API_TOKEN=KGAT_209f2e6f5fd3bc6c81588500c431b21c && kaggle competitions submit -c mallorn-astronomical-classification-challenge -f submission_transformer_ensemble.csv -m "Transformer ensemble CV F1=0.19"
```
