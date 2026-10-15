"""Mark successful launcher admission before entering the existing door."""
import os
from pathlib import Path
import sys
from gpuwm.cycle.ledger import write_receipt_atomically

if __name__ == "__main__":
    marker = Path(sys.argv[1])
    write_receipt_atomically(marker, {"admitted": True, "cards": os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")})
    os.execvpe(sys.argv[2], sys.argv[2:], os.environ)
