# SW-Generation-w-Tooling

### Testing Commands (Processes 2 Valid Questions Each)

These commands will run the evaluation on the first 2 questions from each dataset that contain both a question and a valid ground truth answer (skipping any invalid rows). They will also clear any existing `_failed_*.csv` log files before starting.

1. **Test MedMCQA:**

   ```bash
   python benchmarker.py -d medmcqa -r 1:100 --clear_logs
   ```
2. **Test MMLU Clinical Knowledge:**

   ```bash
   python benchmarker.py -d mmlu_clinical_knowledge -r 1:100 --clear_logs
   ```
3. **Test PubMedQA Labeled (pqa_l):**

   ```bash
   python benchmarker.py -d pubmedqa_pqa_l -r 1:100 --clear_logs
   ```

**(Optional) Test PubMedQA Artificial:**
    ``bash     python benchmarker.py -d pubmedqa_pqa_artificial -n 2 --clear_logs     ``

**Explanation:**

* `python benchmarker.py`: Executes the script.
* `-d [dataset_key]`: Specifies the dataset configuration to use (e.g., `medmcqa`, `pubmedqa_pqa_l`).
* `--range [start:end]`: Sets the index range for evaluation. The start index is inclusive, and the end index is also inclusive. If end is not provided after the colon, the evaluation will process through the end of the dataset. For all data you need to configure it to 0:
* `--clear_logs`: An optional flag added to the script that deletes existing `gemini_failed_...`, `gemma_failed_...`, and `nvidia_failed_...` CSV files for the specified dataset before the run starts, ensuring a fresh log.
