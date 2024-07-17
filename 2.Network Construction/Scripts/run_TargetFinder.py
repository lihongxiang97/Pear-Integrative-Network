import sys
import subprocess
with open(sys.argv[1]) as f:
    lines = f.read().split('>')[1:]
    for line in lines:
        l = line.split('\n')
        id = l[0]
        seq = l[1]
        subprocess.call('perl /storage2/lihongxiang/Project/1.Network-v1/13.sRNA/miRDP2/8.TargetFinder/TargetFinder/targetfinder.pl -s '+
                        seq+' -d '+sys.argv[2]+' -q '+id+' -p table'+' > '+id+'_predicted_targets.txt', shell=True)

