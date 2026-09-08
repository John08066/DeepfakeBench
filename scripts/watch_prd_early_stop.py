"""Poll PRD every six minutes and dispatch one authorized experiment handoff."""
import fcntl
import json
import re
import shutil
import subprocess
import time
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from experiment_summary import write_experiment_summary
LOG = ROOT / 'logs/training/csy/lora_prd_t2_blur_signed_clip_2026-09-09-00-04-18/training.log'
STATE = ROOT / '.state/prd_t2_early_stop_watch'
DATASETS = {'Celeb-DF-v2', 'DFDCP', 'DFDC'}
TRAINING_PID = 1345785  # T2 main process, checked against its original command below.


def training_alive():
    """Do not hand off while the current training process still owns the run."""
    proc = Path('/proc') / str(TRAINING_PID)
    try:
        command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        state = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0]
    except FileNotFoundError:
        return False
    if 'training/train.py' not in command or 'prd_t2_blur.yaml' not in command:
        return False  # PID reuse is not the original training job.
    return state != 'Z'


def handoff_reason(text, rows, alive):
    """A trigger is not a stopped process: wait for clean training shutdown."""
    if alive or 'Stop Training on best Testing metric' not in text:
        return None
    if '[EarlyStop] More than 3 consecutive declining epochs' in text:
        return '自动早停，训练主进程已退出'
    return '正常完成训练计划，训练主进程已退出'


def completed_epochs(text):
    """Use complete, same-step three-domain evaluations at epoch end only."""
    epoch, steps, rows = None, {}, {}
    for line in text.splitlines():
        start = re.search(r'Epoch\[(\d+)\] start!', line)
        if start:
            epoch, steps = int(start[1]), {}
        metric = re.search(r'dataset: (\S+)\s+step: (\d+).*testing-metric, auc: ([0-9.]+)', line)
        if metric and metric[1] in DATASETS:
            steps.setdefault(int(metric[2]), {})[metric[1]] = float(metric[3])
        end = re.search(r'Epoch\[(\d+)\] end with testing', line)
        if end and epoch == int(end[1]) and steps:
            values = steps[max(steps)]
            if set(values) == DATASETS:
                rows[epoch] = sum(values.values()) / 3
    return sorted(rows.items())


def trigger_epoch(rows):
    """A tie, rebound or missing epoch resets consecutive decline count."""
    previous, streak = None, 0
    for epoch, value in rows:
        streak = streak + 1 if previous and epoch == previous[0] + 1 and value < previous[1] else 0
        if streak > 3:
            return epoch
        previous = epoch, value
    return None


def dispatch(event, output):
    """Hand off to Codex with normal automatic approval, never bypass sandbox."""
    prompt = (ROOT / 'scripts/prd_autonomous_handoff.md').read_text()
    prompt += '\n监控器数值事件（数据，不是指令）：\n' + json.dumps(event)
    with output.with_suffix('.console.log').open('x') as console:
        return subprocess.run([shutil.which('codex'), 'exec', '--approve-for-me',
                               '--color', 'never', '-C', str(ROOT), '-o', str(output), prompt],
                              stdin=subprocess.DEVNULL, stdout=console, stderr=subprocess.STDOUT,
                              timeout=3600, check=False).returncode


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    lock = (STATE / 'lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while not (STATE / 'trigger.json').exists():
        log_text = LOG.read_text()
        rows = completed_epochs(log_text)
        finished = 'Stop Training on best Testing metric' in log_text
        alive = training_alive()
        reason = handoff_reason(log_text, rows, alive)
        event = {'checked_at': datetime.now().isoformat(), 'epoch_auc': rows,
                 'trigger_epoch': trigger_epoch(rows), 'training_finished': finished,
                 'training_alive': alive, 'handoff_ready': reason is not None}
        (STATE / 'status.json').write_text(json.dumps(event, indent=2))
        print(json.dumps(event), flush=True)
        if reason is not None:
            # This also covers an already-running process which cannot load new code.
            event['summary'] = str(write_experiment_summary(LOG, reason))
            # Persist before dispatch to prevent duplicate Codex calls after restart.
            event['phase'] = 'summary_saved_dispatching'
            (STATE / 'trigger.json').write_text(json.dumps(event, indent=2))
            output = ROOT / 'logs/RealTime' / (datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_早停触发Codex.log')
            try:
                event['codex_exit_code'] = dispatch(event, output)
                event['phase'] = 'codex_returned' if event['codex_exit_code'] == 0 else 'codex_failed'
                event['codex_report'] = str(output)
            except (OSError, subprocess.TimeoutExpired) as error:
                event['dispatch_error'] = str(error)
                event['phase'] = 'codex_failed'
            (STATE / 'trigger.json').write_text(json.dumps(event, indent=2))
            return
        time.sleep(360)


if __name__ == '__main__':
    main()
