import os
import time

def rewrite_generation(seq, run):
    script = open('generate_data.sh', 'r')
    lines = script.readlines()

    lines[14] = f'python -u cool_model/generate_data.py {run} {seq} >> scripts/logs/run{run:02d}_seq{seq:02d}.out \n'
    print(lines[14])
    with open('generate_data.sh', 'w') as f:
        f.writelines(lines)
        f.close()
    script.close()

for run in range(0, 1):
    if run == 0:
        sequences = range(0,21)
    elif run == 1:
        sequences = os.listdir('/mnt/beegfs/gpu/temporary/vacekpa2/once/data/')

    for seq in sequences:
        #TODO Testovaci nemaji label - proto to nefunguje - zmenit v dataset classe
        rewrite_generation(seq=seq, run=run)
        time.sleep(0.1)
        os.system('sbatch generate_data.sh')



