"""Write-only GFDRV stage probe using actual captured model arguments.

All array files stay on the executing box. Only word-hash receipts travel.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re


def after_call(source, name, occurrence, block):
    starts = list(re.finditer(r"(?im)^\s*call " + name + r"\(", source))
    start = starts[occurrence].start()
    lines = source[start:].splitlines(keepends=True)
    stop = start
    for line in lines:
        stop += len(line)
        code = line.split('!')[0].rstrip()
        if code and not code.endswith('&'):
            return source[:stop] + block + source[stop:]
    raise ValueError(name)


def dumps(names):
    return ''.join(f"      call gftrace_r('{label}',reshape({expr},[size({expr})]))\n"
                   for label, expr in names.items())


def hook(root):
    root = Path(root)
    path = root / 'module_cu_gf_deep.F'
    source = path.read_text()
    source = source.replace('contains', '''integer :: gf_trace_col = 0
contains
subroutine gftrace_r(label, values)
  character(*), intent(in) :: label
  real, intent(in) :: values(:)
  character(1024) :: directory, filename
  integer :: unit, status
  if (gf_trace_col /= 1) return
  call get_environment_variable('GF_TRACE_DIR',directory,status=status)
  if (status /= 0 .or. len_trim(directory) == 0) return
  filename=trim(directory)//'/'//label//'.bin'
  open(newunit=unit,file=trim(filename),access='stream',form='unformatted',status='replace')
  write(unit) values
  close(unit)
end subroutine gftrace_r
''', 1)
    source = source.replace('      flux_tun(:)=fluxtune', dumps({
        'prep_'+n: f'{n}(its:itf,kts:ktf)' for n in
        ('zo','t','q','tn','qo','po','rho','omeg')}) + '      flux_tun(:)=fluxtune',1)
    source = after_call(source,'cup_env',1,dumps({n:f'{n}(its:itf,kts:ktf)' for n in
        ('qes','he','hes','qeso','heo','heso')}))
    source = after_call(source,'cup_env_clev',1,dumps({n:f'{n}(its:itf,kts:ktf)' for n in
        ('qes_cup','q_cup','he_cup','hes_cup','t_cup','qeso_cup','qo_cup',
         'heo_cup','heso_cup','zo_cup','po_cup','gammao_cup','tn_cup')}) +
        dumps({n:f'{n}(its:itf)' for n in ('zws','ztexec','zqexec','cap_max','entr_rate','sig')}))
    source = after_call(source,'rates_up_pdf',1,dumps({
        'zu_pdf':'zuo(its:itf,kts:ktf)', 'entr2d_a':'entr_rate_2d(its:itf,kts:ktf)'}))
    source = after_call(source,'cup_forcing_ens_3d',0,dumps({
        **{n:f'{n}(its:itf)' for n in ('aa0','aa1','tau_ecmwf','aa1_bl','edto','closure_n','xf_dicycle')},
        'mconv2':'mconv(its:itf)',
        **{f'f{k}':f'forcing(its:itf,{k}:{k})' for k in range(1,11)},
        **{n:f'{n}(its:itf,kts:ktf)' for n in ('dellat','dellaq','dellaqc','pwo','pwdo','xhe','xhes','xt','xq')},
    }))
    source = after_call(source,'cup_output_ens_3d',0,dumps({
        **{n+'_o':f'{n}(its:itf,kts:ktf)' for n in ('outt','outq','outqc')},
        'xmb':'xmb(its:itf)',
        **{f'xf{k}':f'xf_ens(its:itf,{k}:{k})' for k in range(1,17)}
    }))
    factor='   fzu = gamma(alpha + beta)/(gamma(alpha)*gamma(beta))'
    if source.count(factor)!=4:
        raise ValueError('the four stock normalization sites are required')
    source=source.replace(factor,factor+'''
   call gftrace_r(trim(draft)//'_tunning',[tunning])
   call gftrace_r(trim(draft)//'_alpha',[alpha])
   call gftrace_r(trim(draft)//'_beta',[beta])
   call gftrace_r(trim(draft)//'_fzu',[fzu])''')
    path.write_text(source)


def wrf_inputs(directory, output, all_columns=False):
    import sys
    import numpy as np
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'wrf_exact_localize'))
    from cmpphys import wrf_array
    root=Path(directory)
    def get(name):
        if name in ('dt','dx','ishallow','clos_choice'):
            raw=(root/f's001_rk1_it00_cu_gf_input__{name}.bin').read_bytes()
            return np.frombuffer(raw[24:], '<i4' if name in ('ishallow','clos_choice') else '<f4')
        shape=(40,40) if name in ('ht','hfx','qfx','xland','kpbl') else (50,40,40)
        value=wrf_array(root,'cu_gf_input',name,shape)
        return value.view(np.int32) if name=='kpbl' else value
    names=('u','v','w','t','qv_curr','p','pi','rho','dz8w','rthften','rqvften','rthraten','rthblten','rqvblten')
    selection=(slice(4,36),slice(4,36)) if all_columns else (slice(4,5),slice(4,5))
    ncol=1024 if all_columns else 1
    profile=lambda n:get(n)[:,selection[0],selection[1]].reshape(50,ncol).T
    levels=np.stack([profile(n) for n in names],axis=1)
    surface=lambda n:np.full(ncol,get(n)[0]) if n in ('dt','dx','ishallow','clos_choice') else get(n)[selection].reshape(ncol)
    sc=np.stack([surface(n) for n in ('dt','dx','ht','hfx','qfx','xland')],axis=1).astype(np.float32)
    ii=np.stack([surface(n) for n in ('kpbl','ishallow','clos_choice')],axis=1).astype(np.int32)
    p8w=np.column_stack((profile('p8w'),np.zeros(ncol,np.float32)))
    with open(output,'wb') as f:
        np.array([47011,ncol,50],np.int32).tofile(f)
        sc.tofile(f); ii.tofile(f); levels.astype(np.float32).tofile(f)
        p8w.astype(np.float32).tofile(f)


def gpu(directory, output, baseline=False):
    import cupy as cp
    import numpy as np
    from gpuwm.core.kernels import module_source_int_defines
    from gf_wrf461_oracle.gf_field_lists import LEV_FIELDS,SCA_FIELDS,ISCA_FIELDS
    root=Path(directory); dst=Path(output); dst.mkdir(exist_ok=True)
    data=np.load(root/'gf_launch_0000.npz')
    inputs=[cp.asarray(data[n][164:165]) for n in ('lvin','scin','iin')]
    source=module_source_int_defines('gf',(('GF_KMAX',50),))
    if baseline:
        # A labelled column diagnostic, never installed in a forecast:
        # measure the old normalizer with the same strict compiler controls.
        old='''return FDIV(gf_wrf_tgamma(FADD(alpha, beta)),
                FMUL(gf_wrf_tgamma(alpha), gfk_tgamma(beta)));'''
        assert source.count(old)==1
        source=source.replace(old,'''return FDIV(gfk_tgamma(FADD(alpha, beta)),
                FMUL(gfk_tgamma(alpha), gfk_tgamma(beta)));''')
    plain=cp.RawModule(code=source,options=('-std=c++17',))
    signature='int k22_wrf_faithful, int n, int nz)'
    start=source.index('void gf_gfdrv_stage(')
    assert source[start:].count(signature)==1
    source=source[:start]+source[start:].replace(signature,'int k22_wrf_faithful, int n, int nz, float *deep_lev, float *deep_sca, int *deep_int, float *prepared)',1)
    anchor='''            kpbl, nz, ichoice,
            GfSink{(float *)0}, GfSink{(float *)0}, GfSinkI{(int *)0},'''
    assert source.count(anchor)==1
    source=source.replace(anchor,'''            kpbl, nz, ichoice,
            GfSink{deep_lev + col*GF_NLEV*nz}, GfSink{deep_sca + col*GF_NSCA}, GfSinkI{deep_int + col*GF_NISCA},''')
    anchor='    float ccn = K_CCN150;'
    block=''
    for i,n in enumerate(('zo','t','q2d','tn_f','qo_f','po','rho','omeg')):
        block+=f'    for(int k=1;k<=nz;k++) prepared[(col*8+{i})*nz+k-1]={n}[k];\n'
    source=source.replace(anchor,block+anchor)
    mod=cp.RawModule(code=source,options=('-std=c++17','--fmad=false','--ftz=false','--prec-div=true','--prec-sqrt=true'))
    outputs=[cp.zeros((1,16,50),cp.float32),cp.zeros((1,9),cp.float32),cp.zeros((1,6),cp.int32)]
    ws=cp.zeros(126*59*64,cp.float32)
    trace=[cp.zeros((1,len(LEV_FIELDS),50),cp.float32),cp.zeros((1,len(SCA_FIELDS)),cp.float32),cp.zeros((1,len(ISCA_FIELDS)),cp.int32),cp.zeros((1,8,50),cp.float32)]
    mod.get_function('gf_gfdrv_stage')((1,),(64,),(*inputs,*outputs,ws,np.int32(0),np.int32(1),np.int32(50),*trace))
    plain_outputs=[cp.zeros_like(a) for a in outputs]
    plain.get_function('gf_gfdrv_stage')((1,),(64,),(*inputs,*plain_outputs,ws,np.int32(0),np.int32(1),np.int32(50)))
    arrays={n:cp.asnumpy(trace[0])[0,i] for i,n in enumerate(LEV_FIELDS)}
    arrays.update({n:cp.asnumpy(trace[1])[0,i:i+1] for i,n in enumerate(SCA_FIELDS)})
    arrays.update({'prep_'+n:cp.asnumpy(trace[3])[0,i] for i,n in enumerate(('zo','t','q','tn','qo','po','rho','omeg'))})
    original=cp.asnumpy(plain_outputs[0])
    actual=cp.asnumpy(outputs[0])
    preserve={'words':int(original.size),'unequal_words':int(np.count_nonzero(original.view(np.uint32)!=actual.view(np.uint32)))}
    (dst/'preservation.json').write_text(json.dumps(preserve))
    arrays.update({n:actual[0,i] for i,n in enumerate(('rthcuten','rqvcuten','rqccuten','rqicuten','dudt','dvdt','gdc','gdc2','outt','outq','outqc','outu','outv','outts','outqs','outqcs'))})
    np.savez(dst/'deep.npz',**arrays)
    receipt(arrays,dst/'receipt.json')


def receipt(arrays,output):
    import numpy as np
    result={}
    for n,a in arrays.items():
        a=np.ascontiguousarray(a,dtype=np.float32).reshape(-1)
        result[n]={'words':a.size,'sha256':hashlib.sha256(a.tobytes()).hexdigest(),
                   'word_sha256':[hashlib.sha256(w.tobytes()).hexdigest() for w in a.view(np.uint32)]}
    Path(output).write_text(json.dumps(result))


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    p=argparse.ArgumentParser(); p.add_argument('action',choices=('hook','wrf-inputs','wrf-inputs-all','gpu','gpu-baseline','receipt'))
    p.add_argument('source'); p.add_argument('output',nargs='?'); a=p.parse_args()
    if a.action=='hook': hook(a.source)
    elif a.action in ('wrf-inputs','wrf-inputs-all'): wrf_inputs(a.source,a.output,a.action=='wrf-inputs-all')
    elif a.action in ('gpu','gpu-baseline'): gpu(a.source,a.output,a.action=='gpu-baseline')
    else:
        import numpy as np
        receipt({f.stem:np.fromfile(f,np.float32) for f in Path(a.source).glob('*.bin')},a.output)
