from pathlib import Path
import os
import shutil
import struct
import subprocess
import sys
import tempfile

from test_guest_intel_trampolines import main_fixture
from test_guest_module_args import args_module, load_and_record
from test_guest_module_directories import module_with_symbol, needed_libraries
from test_guest_needed_modules import executable_with_needed, module_with_needed


def page_aligned(image):
    aligned = bytearray(0x3000)
    aligned[:0x400] = image[:0x400]
    for index in range(3):
        offset = 64 + index * 56
        header = list(struct.unpack_from("<IIQQQQQQ", image, offset))
        aligned[header[3]:header[3] + header[5]] = image[header[2]:header[2] + header[5]]
        header[2] = header[3]
        if header[0] == 1:
            header[7] = 0x1000
        struct.pack_into("<IIQQQQQQ", aligned, offset, *header)
    return aligned


def main():
    relinker = Path(sys.argv[1]).resolve()
    host = Path(sys.argv[2]).resolve()
    with tempfile.TemporaryDirectory(prefix="anyps5-deferred-dependencies-") as directory:
        for root in ("runtime", "executable", "bundled"):
            case = Path(directory) / root
            bundled = case / "sce_module"
            plugins = case / "Media" / "Plugins"
            bundled.mkdir(parents=True)
            plugins.mkdir(parents=True)
            (plugins / "provider.prx").write_bytes(args_module())
            (plugins / "consumer.prx").write_bytes(page_aligned(module_with_symbol(False)))
            (plugins / "plugin.prx").write_bytes(page_aligned(module_with_needed((b"consumer.prx",), (0,))))
            if root == "bundled":
                (bundled / "root.prx").write_bytes(page_aligned(module_with_needed((b"plugin.prx",), (0,))))
            source = case / "input.elf"
            executable = executable_with_needed(b"plugin.prx") if root == "executable" else main_fixture()
            code = bytes.fromhex("b83c000000bf2a0000000f05")
            executable[0x4000:0x4000 + len(code)] = code
            source.write_bytes(executable)
            output = case / "output.elf"
            result = subprocess.run([str(relinker), "--skip-syscall-check", "--module-dir", "Media/Plugins", str(source), str(output)],
                                    capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, (root, result.stdout, result.stderr)
            needed = needed_libraries(output.read_bytes())
            runtime = case / "app0" / "Media" / "Plugins"
            for name in ("plugin", "consumer", "provider"):
                dependency = f"$ORIGIN/app0/Media/Plugins/{name}.prx.guest.prx"
                assert (dependency in needed) == (root != "runtime"), (root, needed)
            assert "$ORIGIN/consumer.prx.guest.prx" in needed_libraries((runtime / "plugin.prx.guest.prx").read_bytes())
            assert "$ORIGIN/provider.prx.guest.prx" in needed_libraries((runtime / "consumer.prx.guest.prx").read_bytes())
            if sys.platform == "linux" and os.uname().machine == "x86_64":
                libraries = case / "libs"
                libraries.mkdir()
                shutil.copyfile(host, libraries / "libc.prx")
                events = case / "startup-events.txt"
                output.chmod(0o755)
                run = subprocess.run([str(output)], capture_output=True, text=True, timeout=30,
                                     env={**os.environ, "ANYPS5_GUEST_MODULE_ARGS": str(events)})
                assert run.returncode == 42, (root, run.returncode, run.stdout, run.stderr)
                recorded = events.read_text() if events.exists() else ""
                assert recorded.count("args=") == (0 if root == "runtime" else 1), (root, recorded)
                load_and_record(case, libraries / "libc.prx", runtime / "plugin.prx.guest.prx")
    print("Guest deferred dependency tests passed")


if __name__ == "__main__":
    main()
