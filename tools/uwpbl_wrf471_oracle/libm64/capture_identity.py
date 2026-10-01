import subprocess,json,hashlib,platform
from pathlib import Path
root=Path(__file__).parent
for label,start,end in [('exp-resolver',0x602b0,0x60310),('log-resolver',0x71ba0,0x71c20),('pow-resolver',0x723b0,0x72430),('public-wrappers',0x5af50,0x5b240),('acos-wrapper',0x17430,0x17487),('acos-resolver',0x5d810,0x5d870)]:
 (root/(label+'.objdump.txt')).write_text(subprocess.run(['objdump','-d',f'--start-address={start}',f'--stop-address={end}','/lib/x86_64-linux-gnu/libm.so.6'],capture_output=True,text=True,check=True).stdout)
info={}
for args in [['ldd','--version'],['gcc','--version'],['objdump','--version'],['lscpu'],['sha256sum','/lib/x86_64-linux-gnu/libm.so.6']]:info[' '.join(args)]=subprocess.run(args,capture_output=True,text=True).stdout
(root/'host-environment.json').write_text(json.dumps(info,indent=2))
(root/'probe.log').write_text(subprocess.run(['python3',str(root/'probe.py')],capture_output=True,text=True,check=True).stdout)
(root/'data-probe.log').write_text(subprocess.run(['python3',str(root/'data_probe.py')],capture_output=True,text=True,check=True).stdout)
