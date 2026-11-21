"""Durable metadata commit after complete shared FULL+RAW writes."""
import json,os
from pathlib import Path

def write_json(path,value,*,replace=False):
    path=Path(path)
    if path.exists()and not replace:raise FileExistsError(path)
    temporary=path.with_name(path.name+'.partial')
    with temporary.open('x')as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    descriptor=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)
