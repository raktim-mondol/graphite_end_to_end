import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

class ProgressTracker:
    """Track progress of pipeline steps and save logs."""

    def __init__(self, output_dir: str = "outputs/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self.step_times: Dict[str, Dict] = {}
        self.progress_log = []

    def start_step(self, step_name: str, description: str = ""):
        self.step_times[step_name] = {'start': time.time()}
        self.progress_log.append({
            'timestamp': datetime.now().isoformat(),
            'step': step_name,
            'action': 'start',
            'description': description
        })
        self._save_progress()

    def complete_step(self, step_name: str, results: Dict = None):
        if step_name in self.step_times:
            self.step_times[step_name]['end'] = time.time()
            self.step_times[step_name]['duration'] = (
                self.step_times[step_name]['end'] -
                self.step_times[step_name]['start']
            )
        self.progress_log.append({
            'timestamp': datetime.now().isoformat(),
            'step': step_name,
            'action': 'complete',
            'duration': self.step_times.get(step_name, {}).get('duration', 0),
            'results': results or {}
        })
        self._save_progress()

    def _save_progress(self):
        progress_file = self.output_dir / "pipeline_progress.json"
        with open(progress_file, 'w') as f:
            json.dump({
                'total_runtime': time.time() - self.start_time,
                'step_times': self.step_times,
                'progress_log': self.progress_log
            }, f, indent=2)
