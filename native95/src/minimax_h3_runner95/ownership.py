"""Runner-owned admission for the exact captured two-UUID H3 profile.

This is a new boundary, not a Root receipt or a portable placement derivative.
The external runner owns policy, monitoring, process/PDEATH and lease lifetime.
The library independently checks actual FD identities and kernel FLOCK modes.
No framework is imported by this module.
"""
from dataclasses import dataclass
from pathlib import Path
import os
import stat
import threading

UUIDS = ('GPU-72c1b506-891d-b8bc-b353-e020585e1c47',
         'GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69')
CAPS = dict(whole_host_GiB=72, joint_RSS_GiB=72,
            GPU_NVML_GiB=(37, 42), GPU_allocator_GiB=(35, 40),
            ENTRY_GPU_free_GiB=(5, 5), LIVE_GPU_free_GiB=(5, 5),
            host_free_floor_GiB=16, all_other_CURRENT_required=True,
            CPU_model_math_or_offload=False, elapsed_deadline=None)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def _birth():
    # /proc stat field22, after the parenthesized comm (which may contain spaces).
    raw = Path('/proc/self/stat').read_text()
    return raw[raw.rfind(')') + 2:].split()[19]


def _lock_modes(text):
    """Only actual FLOCK advisory lock rows; no filename/FD existence inference."""
    rows = []
    for line in text.splitlines():
        fields = line.split()
        if fields[:1] == ['lock:']:
            require(len(fields) >= 8 and fields[2:4] == ['FLOCK', 'ADVISORY'],
                    'An actual advisory FLOCK descriptor is required')
            rows.append(fields[4])
    return rows


@dataclass(frozen=True)
class HeldLease:
    path: str
    fd: int
    mode: str

    def check(self):
        require(type(self.fd) is int and self.fd >= 0 and self.mode in ('SH', 'EX'),
                'Literal held lease FD and SH/EX mode required')
        path = Path(self.path)
        require(path.is_absolute() and '..' not in path.parts and not path.is_symlink(),
                'Literal nonsymlink lease path required')
        opened, named = os.fstat(self.fd), path.stat()
        require(stat.S_ISREG(opened.st_mode)
                and (opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino),
                'Held lease FD no longer names the original regular file')
        modes = _lock_modes(Path('/proc/self/fdinfo/' + str(self.fd)).read_text())
        require(modes == [('READ' if self.mode == 'SH' else 'WRITE')],
                'Actual kernel FLOCK mode differs from the declared lease')


@dataclass(frozen=True)
class RunnerAdmission:
    owner_id: str
    physical_uuids: tuple
    physical_shared: tuple
    role_exclusive: HeldLease
    check_live: object

    def structural_check(self):
        require(isinstance(self.owner_id, str) and bool(self.owner_id.strip())
                and self.physical_uuids == UUIDS and callable(self.check_live),
                'Explicit runner identity, captured physical UUIDs and policy checker required')
        require(type(self.physical_shared) is tuple and len(self.physical_shared) == 4,
                'Two research/ordinary shared leases for each captured GPU required')
        leases = (*self.physical_shared, self.role_exclusive)
        require(all(type(x) is HeldLease for x in leases)
                and [x.mode for x in leases] == ['SH', 'SH', 'SH', 'SH', 'EX']
                and len({x.fd for x in leases}) == len({x.path for x in leases}) == 5,
                'Five distinct actual SH/SH/SH/SH/EX lease descriptors required')
        for lease in leases:
            lease.check()
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == ','.join(UUIDS),
                'The runner must select the exact captured physical GPU mask before entry')

    def check(self):
        self.structural_check()
        # The runner MUST implement this with actual policy + current telemetry,
        # all other CURRENT forecasts, host/device caps/floors, both GPU lease
        # mappings, SH/EX modes and real parent/PDEATH/lifetime verification.
        # It must also monitor throughout long native calls. A callback or FD
        # alone is not a resource reservation, physical fit or admission proof.
        require(self.check_live(CAPS.copy(), self) is True,
                'The actual runner policy/lifetime check refused this operation')


class OwnerIdentity:
    """One process/thread lifetime; never borrowed by another worker or thread."""
    def __init__(self):
        self.pid, self.birth, self.thread = os.getpid(), _birth(), threading.get_ident()

    def check(self):
        require((os.getpid(), _birth(), threading.get_ident())
                == (self.pid, self.birth, self.thread),
                'The admitted runner scope belongs to a different process/thread lifetime')
