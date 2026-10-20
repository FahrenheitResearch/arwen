"""Read logical receipt bytes from their original or lossless gzip file."""
import gzip

def receipt_bytes(path):
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return gzip.decompress(path.with_suffix(path.suffix+'.gz').read_bytes())
