# End-to-End Pipeline Testing Strategy

## 1. Overview of End-to-End Pipeline Testing

This document outlines the testing strategy for the end-to-end pipeline orchestration script (`main_pipeline.py`). The `main_pipeline.py` script is responsible for coordinating the sequential execution of the different stages of the GRAPHITE pipeline:

1.  **MIL Classification**
2.  **Self-Supervised Learning (SSL)**
3.  **Explainable AI (XAI) Visualization**
4.  **Saliency Map Fusion**

The unit tests for `main_pipeline.py` are designed to ensure the robustness and correctness of this orchestration logic. They verify that the pipeline controller correctly initializes and calls each step, manages data flow, handles configurations, and processes command-line arguments as expected.

**Important Note:** These tests focus on the pipeline's orchestration capabilities. The individual computational steps (e.g., model training within MILStep or SSLStep) are mocked. This means these unit tests do *not* execute the actual machine learning model training or complex data processing of each full pipeline stage. They verify the pipeline's control flow.

## 2. Test Setup & Prerequisites

### Configuration
The tests use a dedicated configuration file:
-   `tests/config/test_pipeline_config.yaml`

This configuration uses minimal parameters (e.g., 1 epoch, batch size of 1) and dummy paths to ensure tests run quickly and do not depend on large datasets or pre-trained models from the full pipeline.

### Environment
-   **Python:** Ensure Python 3.9+ is installed.
-   **Dependencies:** The necessary Python packages are listed in the main `requirements.txt` file in the project root. Install them using:
    ```bash
    pip install -r requirements.txt
    ```
    Key libraries used by the tests themselves include `unittest` (standard library) and `PyYAML` (for config loading). The `main_pipeline.py` script and its components might require additional libraries like `torch`, which should also be installed via `requirements.txt`.

### Test Data
The tests are designed to run without specific large-scale datasets. The `paths.data_root` in `test_pipeline_config.yaml` points to `tests/data/`, which is expected to be an empty directory or contain minimal placeholder files if any step's initialization (even mocked) requires file/directory existence checks.

## 3. How to Run the Tests

To execute the end-to-end pipeline unit tests, navigate to the project's root directory and run the following command:

```bash
python -m unittest tests.test_main_pipeline
```

Alternatively, if you have a test runner integrated into your IDE, you can typically run tests by right-clicking on the `tests/test_main_pipeline.py` file or the `TestMainPipeline` class.

## 4. Test Case Summary

The test suite (`tests/test_main_pipeline.py`) includes the following key verification areas:

-   **Successful Full Pipeline Execution:**
    -   Verifies that `main_pipeline.py` can run all defined steps (MIL, SSL, XAI, Fusion) in sequence when no specific steps are provided via arguments.
    -   Ensures helper classes (`DataFlowManager`, `ModelManager`, `ProgressTracker`) are initialized correctly.
    -   Checks that a final pipeline report is generated.
-   **Execution of Specific Steps:**
    -   Tests the `--steps` command-line argument, ensuring that only the specified steps are executed (e.g., only `mil` and `ssl`).
-   **Configuration Loading:**
    -   Verifies that the `load_config` function correctly loads and parses the YAML configuration file.
-   **Report Generation:**
    -   Tests the `generate_pipeline_report` function to ensure it correctly writes the results to a YAML file.
-   **Error Handling:**
    -   **Invalid Configuration Path:** Ensures the pipeline exits gracefully or raises an appropriate error if the specified configuration file path is invalid.
    -   **Failing Pipeline Step:** Verifies that if a step within the pipeline raises an exception, this failure is handled (e.g., the exception propagates, and subsequent steps are not executed).

## 5. How to Interpret Results

When you run the tests, `unittest` will output information to the console:

-   **Dots (`.`):** Each dot usually represents a test case that passed.
-   **`F`:** Indicates a test case that failed due to an assertion error (the test's expectation was not met).
-   **`E`:** Indicates a test case that encountered an unexpected error (e.g., an unhandled exception in the code being tested).

At the end of the output, a summary will be provided, stating the total number of tests run and the number of failures or errors.

Example of a successful run:
```
.....
----------------------------------------------------------------------
Ran 5 tests in 0.010s

OK
```

Example of a run with failures:
```
..F..E.
======================================================================
FAIL: test_example_failure (tests.test_main_pipeline.TestMainPipeline)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/path/to/project/tests/test_main_pipeline.py", line XX, in test_example_failure
    self.assertEqual(a, b)
AssertionError: X != Y

======================================================================
ERROR: test_example_error (tests.test_main_pipeline.TestMainPipeline)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/path/to/project/main_pipeline.py", line YY, in some_function
    raise ValueError("Something went wrong")
ValueError: Something went wrong

----------------------------------------------------------------------
Ran 7 tests in 0.015s

FAILED (failures=1, errors=1)
```
If failures or errors occur, the traceback will help pinpoint the issue in either the test code or the source code.

## 6. Mocking Strategy

As mentioned in the Overview, these unit tests employ a mocking strategy using Python's `unittest.mock` module (specifically `@patch` decorators).

-   The `execute()` method of each main pipeline step (`MILStep`, `SSLStep`, `XAIStep`, `FusionStep`) is replaced with a mock object.
-   This means that when `main_pipeline.py` calls `step.execute()`, the actual complex logic of that step is *not* run. Instead, the mock records the call and returns a predefined value or simulates a predefined behavior (like raising an exception for error handling tests).
-   Helper classes like `DataFlowManager`, `ModelManager`, and `ProgressTracker` are also mocked in some tests to verify they are initialized as expected by `main_pipeline.py` without performing actual file I/O or model loading.

This approach allows the tests to be:
-   **Fast:** They don't wait for lengthy computations.
-   **Focused:** They specifically test the orchestration logic of `main_pipeline.py`.
-   **Isolated:** Failures in a specific step's complex internal logic (which should be covered by its own dedicated unit tests) do not cause these pipeline orchestration tests to fail, unless the failure is in the interaction contract between the step and the pipeline.

Researchers wishing to reproduce or understand the full pipeline's behavior (including the ML components) should run the `main_pipeline.py` script with appropriate configurations and data, as described in `END_TO_END_PIPELINE.md` or the main project `README.md`. These unit tests serve to validate the control software that ties the steps together.
