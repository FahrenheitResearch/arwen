"""The stock namelist door reaches the generation and source table controls."""
import subprocess,sys,tomllib
from pathlib import Path
from test_urban_registry import _generic_hierarchy_pair


def test_real_cli_import_selects_the_vegetated_arm(tmp_path):
    wps,inp=_generic_hierarchy_pair(tmp_path,1)
    inp.write_text(inp.read_text().replace('&physics','&physics\n sf_urban_physics=1,',1))
    out=tmp_path/'resolved.toml'
    run=subprocess.run([sys.executable,'-m','gpuwm.cli','import-namelist',str(wps),str(inp),
        '--urban-ucm-generation','wrf_480','--urban-treeoption','1','--urban-ttscheme','1','--urban-fvg','.2','--output',str(out)],text=True,capture_output=True)
    assert run.returncode==0,run.stdout+run.stderr
    shared=tomllib.loads(out.read_text())['shared']
    assert shared['sf_urban_physics']==1
    assert shared['urban_ucm_generation']=='wrf_480'
    assert shared['urban_treeoption']==1
    assert shared['urban_ttscheme']==1
    assert shared['urban_fvg']==.2


def test_supervised_worker_parser_retains_the_same_controls():
    from gpuwm.wrfinput_forecast import build_parser
    args=build_parser().parse_args(['--wrfinput','input','--outdir','out','--urban-ucm-generation','wrf_480','--urban-treeoption','1','--urban-fvg','.2'])
    assert (args.urban_ucm_generation,args.urban_treeoption,args.urban_fvg)==('wrf_480',1,.2)
