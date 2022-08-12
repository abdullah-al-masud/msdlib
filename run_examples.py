import subprocess
import os
import time


example_directory = 'examples'
skip_list = [
    'filters_and_spectrogram_example.py',
    'ProgressBar_example.py',
    'torchModel_with_conv_model_example.py'
]

def run_example_scripts(example_dir, skip_list):

    t = time.time()
    failed = []
    passed = []
    for fname in os.listdir(example_dir):
        if fname.endswith('.py') and fname not in skip_list:
            print('Running %s... '%fname, end='', flush=True)
            fpath = os.path.join(example_dir, fname)
            command = 'python %s'%(fpath)
            out = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            print('   complete, return code', out.returncode, '.')
            
            if out.returncode != 0:
                failed.append(fname)
            else:
                passed.append(fname)
    fail_msg = 'All examples ran successfully' if len(failed) == 0 else '\n\nFailed scripts are: \n' + str(failed)
    print(fail_msg)
    print('Skipped these scripts:\n%s'%skip_list)
    print('Total: %d,  skipped: %d,  success: %d,  failed: %d'%(len(passed) + len(skip_list) + len(failed),
                                                                   len(skip_list), len(passed), len(failed)))
    print('total elapsed time: %.2f minutes.' %((time.time() - t) / 60))

if __name__ == "__main__":
    run_example_scripts(example_directory, skip_list)