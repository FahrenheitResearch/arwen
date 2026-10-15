"""Make auditable CUDA from gfortran's -O0 lowering of pinned WRF Lin.

Only the default six-species arm is emitted. This closed development tool
refuses unknown lowering syntax. The compiled WRF oracle remains unmodified.
The compiler lowering fixes constant folding, real powers, and operation order.
No compiler or source tree is needed at runtime.
"""
from pathlib import Path
import argparse
import hashlib
import re

PIN = 'd66e442fccc04111067e29274c9f9eaccc3cef28'
NAMES = ('ggamma', 'parama1', 'parama2', 'satadj', 'clphy1d')


def block_end(s, start):
    assert s[start] == '{'
    depth = 1
    p = start + 1
    while depth:
        depth += (s[p] == '{') - (s[p] == '}')
        p += 1
    return p


def default_drop_arm(s):
    """WRF's F_QNDROP=false arm, not a substitute number-moment scheme."""
    while (m := re.search(r'if \(\*flag_qndrop\)\s*\{', s)):
        a = s.index('{', m.start())
        b = block_end(s, a)
        tail = re.match(r'\s*else\s*\{', s[b:])
        if tail:
            c = s.index('{', b)
            d = block_end(s, c)
            s = s[:m.start()] + s[c+1:d-1] + s[d:]
        else:
            s = s[:m.start()] + s[b:]
    return s


def expression_nodes(s):
    names = {'MAX_EXPR':'lin_max', 'MIN_EXPR':'lin_min',
             'ABS_EXPR':'fabsf', 'NON_LVALUE_EXPR':''}
    while (m := re.search(r'(\w+_EXPR) <', s)):
        name = m[1]
        if name not in names:
            raise ValueError(('unknown lowered node', name))
        p = m.end()
        depth = 1
        while depth:
            depth += (s[p] == '<') - (s[p] == '>')
            p += 1
        arg = expression_nodes(s[m.end():p-1])
        s = s[:m.start()] + names[name] + '(' + arg + ')' + s[p:]
    return s


def normalize(s):
    s = expression_nodes(s)
    s = re.sub(r'\+ offset\.\d+', '', s)
    s = re.sub(r'\(\*(\w+)\)\[', r'\1[', s)
    s = re.sub(r'real\(kind=4\)\[0:D\.\d+\] \* restrict (\w+)', r'float* \1', s)
    s = s.replace('real(kind=4) & restrict ', 'float* ')
    s = s.replace('integer(kind=4) & restrict ', 'const int* ')
    s = s.replace('logical(kind=4) & restrict ', 'const bool* ')
    s = s.replace('real(kind=4)', 'float').replace('real(kind=8)', 'double')
    s = s.replace('integer(kind=4)', 'int').replace('integer(kind=8)', 'long long')
    s = s.replace('logical(kind=4)', 'bool')
    s = s.replace(' restrict', '')
    s = re.sub(r'\b([DCML])\.(\d+)', r'\1_\2', s)
    for name in NAMES:
        s = re.sub(r'\b' + name + r'\b', 'lin_' + name, s)
    s = s.replace('__builtin_powif', 'lin_powi')
    s = s.replace('__builtin_powf', 'gfk_pow')
    s = s.replace('__builtin_expf', 'gfk_exp')
    s = s.replace('__builtin_sqrtf', 'sqrtf')
    s = re.sub(r'(?<![\w.])(?:\d+\.\d*(?:e[+-]?\d+)?|\d+e[+-]?\d+)(?![\w.])', lambda m: m[0]+'f', s)
    s = s.replace('8.0e+6f', 'LIN_XNOR')
    s = s.replace('static float', 'const float')
    s = s.replace('lin_ggamma (float* x)', 'lin_ggamma (const float* x)')
    s = s.replace('lin_parama1 (float* temp)', 'lin_parama1 (const float* temp)')
    s = s.replace('lin_parama2 (float* temp)', 'lin_parama2 (const float* temp)')
    if any(x in s for x in ('kind=', 'offset.', '__builtin_', 'finally', 'restrict', '_EXPR', '0B')):
        raise ValueError('unhandled compiler syntax')
    return s


def generate(dump):
    functions = []
    for name in NAMES:
        m = re.search(r'^(?:void|real\(kind=4\)) ' + name + r' \(', dump, re.M)
        assert m, name
        a = dump.index('\n{', m.start()) + 1
        b = block_end(dump, a)
        signature = dump[m.start():a]
        raw = dump[a+1:b-1]
        if name in ('parama1','parama2','ggamma'):
            if name == 'ggamma':
                aa = raw.index('  L.28:;') + len('  L.28:;')
                bb = raw.index('  __label_000020:;')
                raw = raw[:aa] + '\n  return 0.0; // unreachable: all internal arguments are <= 6.8\n' + raw[bb:]
            body = raw
        else:
            top = raw.split('\n  try', 1)[0]
            declarations = []
            for line in top.splitlines():
                arr = re.fullmatch(r'  real\(kind=4\)\[0:D\.\d+\] \* restrict (\w+);', line)
                if arr:
                    declarations.append('  float ' + arr[1] + '[LIN_KMAX+1];')
                elif re.match(r'  (?:static )?(?:real|integer|logical)\(kind=4\) \w+(?: = [^;]+)?;', line):
                    if not any(x in line for x in (' ncall', ' D.', ' offset.', ' size.', ' lbound.', ' ubound.')):
                        declarations.append(line)
            start = raw.index('      dtb = *dt;' if name == 'clphy1d' else '      (*thz)[')
            end = raw.index('\n    }\n  finally')
            body = '\n'.join(declarations) + '\n' + default_drop_arm(raw[start:end])
        functions.append('__device__ ' + normalize(signature) + '{\n' + normalize(body) + '\n}\n')
    return '\n'.join(functions)


def pin_operations(source):
    # CUDA can replace division by a known denominator with reciprocal
    # multiplication. Emit explicit IEEE operations for the strict arm.
    from pycparser import c_parser, c_ast, c_generator
    plain = source.replace('__device__ ', '')
    plain = re.sub(r'//[^\n]*', '', plain)
    tree = c_parser.CParser().parse('typedef int bool;\n' + plain)
    for function in tree.ext[1:]:
        types = {'LIN_XNOR':'float'}
        class Declarations(c_ast.NodeVisitor):
            def visit_Decl(self, node):
                base = node.type
                while hasattr(base, 'type'):
                    base = base.type
                if hasattr(base, 'names'):
                    types[node.name] = ' '.join(base.names)
                self.generic_visit(node)
        Declarations().visit(function)
        def kind(node):
            if isinstance(node, c_ast.Constant): return node.type
            if isinstance(node, c_ast.ID): return types.get(node.name, 'int')
            if isinstance(node, (c_ast.ArrayRef, c_ast.UnaryOp)):
                return kind(node.name if isinstance(node,c_ast.ArrayRef) else node.expr)
            if isinstance(node, c_ast.Cast):
                base=node.to_type.type
                while hasattr(base, 'type'): base=base.type
                return ' '.join(base.names)
            if isinstance(node,c_ast.BinaryOp):
                return 'float' if 'float' in (kind(node.left),kind(node.right)) else 'int'
            if isinstance(node,c_ast.FuncCall):
                if node.name.name in ('lin_min','lin_max'): return kind(node.args.exprs[0])
                return 'float'
            raise ValueError(('expression type',type(node)))
        def walk(node):
            for attr, child in node.children():
                new = walk(child)
                if '[' in attr:
                    field,index=attr[:-1].split('[')
                    getattr(node,field)[int(index)]=new
                else: setattr(node,attr,new)
            if isinstance(node,c_ast.BinaryOp) and node.op in ('+','-','*','/') and kind(node)=='float':
                fn={'+':'lin_add','-':'lin_sub','*':'lin_mul','/':'lin_div'}[node.op]
                return c_ast.FuncCall(c_ast.ID(fn),c_ast.ExprList([node.left,node.right]))
            return node
        walk(function)
    generated=c_generator.CGenerator().visit(tree)
    generated=generated[generated.index('float lin_ggamma'):]
    generated=re.sub(r'^(float|void) lin_',r'__device__ \1 lin_',generated,flags=re.M)
    return generated + '\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dump', type=Path)
    p.add_argument('wrf_source', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    expected = "bb9d0b99ab4ecd5ed6ce566e20f8f0b74b496410f47b4a9211a51501224f4bb69"
    if hashlib.sha256(a.wrf_source.read_bytes()).hexdigest() != expected:
        raise ValueError("WRF Lin source differs from the pinned oracle source")
    header = '// Generated by tools/lin_wrf461_oracle/transcribe.py.\n'
    header += '// WRF v4.6.1 phys/module_mp_lin.F, ' + PIN + '\n'
    header += '// Source SHA256: ' + hashlib.sha256(a.wrf_source.read_bytes()).hexdigest() + '\n'
    header += '// gfortran 13 -O0 lowering; default F_QNDROP=false.\n'
    header += '// Original statement order and rounded constant words are retained.\n'
    a.output.write_text(header + pin_operations(generate(a.dump.read_text())), encoding='utf-8')


if __name__ == '__main__':
    main()
