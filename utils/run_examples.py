"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""

import os
import subprocess
import sys
import time


example_directory = 'examples'
skip_list = [
    'filters_and_spectrogram_example.py',
    'ProgressBar_example.py',
    'torchModel_with_conv_model_example.py',
]


def run_example_scripts(example_dir, skip_list):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    example_dir = os.path.join(project_root, example_dir)
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(
        [project_root] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else [])
    )
    env.setdefault('MPLBACKEND', 'Agg')

    t = time.time()
    failed = []
    passed = []
    for fname in sorted(os.listdir(example_dir)):
        if fname.endswith('.py') and fname not in skip_list:
            print('Running %s... ' % fname, end='', flush=True)
            fpath = os.path.join(example_dir, fname)
            out = subprocess.run(
                [sys.executable, fpath],
                cwd=project_root,
                env=env,
                capture_output=True,
                text=True,
            )
            print('   complete, return code', out.returncode, '.')

            if out.returncode != 0:
                failed.append(fname)
                err_tail = (out.stderr or out.stdout or '').strip().splitlines()
                if err_tail:
                    print('   error:', err_tail[-1])
            else:
                passed.append(fname)
    fail_msg = 'All examples ran successfully' if len(failed) == 0 else '\n\nFailed scripts are: \n' + str(failed)
    print(fail_msg)
    print('Skipped these scripts:\n%s' % skip_list)
    print('Total: %d,  skipped: %d,  success: %d,  failed: %d' % (
        len(passed) + len(skip_list) + len(failed),
        len(skip_list), len(passed), len(failed)))
    print('total elapsed time: %.2f minutes.' % ((time.time() - t) / 60))


if __name__ == "__main__":
    run_example_scripts(example_directory, skip_list)
