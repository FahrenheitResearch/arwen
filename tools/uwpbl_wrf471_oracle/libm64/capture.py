import subprocess,pathlib,json,ctypes,struct
root=pathlib.Path(__file__).parent
for label,start,end in [('exp',0xa0b50,0xa0d38),('log',0xa11c0,0xa13fa),('pow',0xa1850,0xa1e91)]:
 p=subprocess.run(['objdump','-d',f'--start-address={start}',f'--stop-address={end}','/lib/x86_64-linux-gnu/libm.so.6'],capture_output=True,text=True)
 (root/(label+'.objdump.txt')).write_text(p.stdout)
print(subprocess.run(['gcc','--version'],capture_output=True,text=True).stdout)
