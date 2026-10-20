"""Read-only evidence at the exact scheduled full-state health checks."""
import hashlib
import cupy as cp
import numpy as np

def array_record(value):
    array=cp.asnumpy(value)
    return dict(shape=list(array.shape),dtype=array.dtype.str,words=int(array.size),sha256=hashlib.sha256(array.tobytes()).hexdigest())

def health_witness(validators):
    cp.cuda.get_current_stream().synchronize()
    result={}
    for gid,validator in sorted(validators.items()):
        fields=validator.fields
        pointers=cp.asnumpy(validator._ptr).view(np.uint64)
        auxiliaries=cp.asnumpy(validator._aux).view(np.uint64)
        expected=[int(field.values.data.ptr) for field in fields]
        expected_aux=[int(field.auxiliary.data.ptr) if field.auxiliary is not None else 0 for field in fields]
        assert pointers[:len(fields)].tolist()==expected and not pointers[len(fields):].any(),('field descriptors differ',gid)
        assert auxiliaries[:len(fields)].tolist()==expected_aux and not auxiliaries[len(fields):].any(),('aux descriptors differ',gid)
        rows=[]
        for field in fields:
            rows.append(dict(name=field.name,rule=repr(field.rule),aux_mode=field.aux_mode,plane_size=field.plane_size,values=array_record(field.values),auxiliary=array_record(field.auxiliary) if field.auxiliary is not None else None))
        result[str(gid)]=dict(fields=rows,result=array_record(validator._result),chunk=validator._chunk,chunk_count=validator._chunk_count,field_pointer_count=len(expected),auxiliary_pointer_count=sum(bool(p) for p in expected_aux))
    return result
