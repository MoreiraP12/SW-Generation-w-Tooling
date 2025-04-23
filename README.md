# SW-Generation-w-Tooling

### Testing Commands (Processes 2 Valid Questions Each)

These commands will run the evaluation on the first 2 questions from each dataset that contain both a question and a valid ground truth answer (skipping any invalid rows). They will also clear any existing `_failed_*.csv` log files before starting.

1. **Test MedMCQA:**

   ```bash
   python benchmarker.py -d medmcqa -n 2 --clear_logs
   ```
2. **Test MMLU Clinical Knowledge:**

   ```bash
   python benchmarker.py -d mmlu_clinical_knowledge -n 2 --clear_logs
   ```
3. **Test PubMedQA Labeled (pqa_l):**

   ```bash
   python benchmarker.py -d pubmedqa_pqa_l -n 2 --clear_logs
   ```

**(Optional) Test PubMedQA Artificial:**
    ``bash     python benchmarker.py -d pubmedqa_pqa_artificial -n 2 --clear_logs     ``

**Explanation:**

* `python benchmarker.py`: Executes the script.
* `-d [dataset_key]`: Specifies the dataset configuration to use (e.g., `medmcqa`, `pubmedqa_pqa_l`).
* `-n 2`: Sets the `max_questions` argument. The script will stop after successfully processing 2 questions that have valid question text and valid ground truth answers (it might check more rows if initial ones are skipped).
* `--clear_logs`: An optional flag added to the script that deletes existing `gemini_failed_...`, `gemma_failed_...`, and `nvidia_failed_...` CSV files for the specified dataset before the run starts, ensuring a fresh log.
