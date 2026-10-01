#!/usr/bin/env python3
"""Stand-in for sglang/negeig_sglang.py in the parity.sh tests. `static-check` exits STUB_STATIC_RC (default 0);
`install-plugin --dir DIR` writes a module and a real-format dist-info (METADATA plus entry_points.txt with
[sglang.srt.plugins] negeig = negeig_sglang:register) so the entry-point discovery check in parity.sh runs for real.
STUB_PLUGIN_NO_ENTRYPOINT=1 writes the entry point under the wrong group (a plugin SGLang would not find);
STUB_INSTALL_RC makes install-plugin fail."""
import os
import sys
from pathlib import Path


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "static-check":
        rc = int(os.environ.get("STUB_STATIC_RC", "0"))
        print("static-check: " + ("ok" if rc == 0 else "FAILED (stub)"))
        return rc
    if cmd == "install-plugin":
        if int(os.environ.get("STUB_INSTALL_RC", "0")):
            print("install-plugin: planned failure", file=sys.stderr)
            return int(os.environ["STUB_INSTALL_RC"])
        d = Path(sys.argv[sys.argv.index("--dir") + 1])
        di = d / "negeig_sglang-1.0.0.dist-info"
        di.mkdir(parents=True, exist_ok=True)
        (d / "negeig_sglang.py").write_text("def register():\n    pass\n")
        (di / "METADATA").write_text("Metadata-Version: 2.1\nName: negeig-sglang\nVersion: 1.0.0\n")
        group = "sglang.srt.plugins" if os.environ.get("STUB_PLUGIN_NO_ENTRYPOINT") != "1" else "not.sglang.plugins"
        (di / "entry_points.txt").write_text(f"[{group}]\nnegeig = negeig_sglang:register\n")
        print(f"install-plugin: wrote {d}")
        return 0
    print(f"fake negeig_sglang: unknown command {cmd!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
