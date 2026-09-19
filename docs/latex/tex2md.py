"""Regenerate docs/paper.md from main.tex so the two cannot drift.

    python docs/latex/tex2md.py        # after `make`, which writes main.aux

Resolves \\cite/\\citet/\\citep from references.bib, \\ref from main.aux, and
renders \\TBD{...} visibly, then hands the body to pandoc (gfm).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "paper.md"


_ACCENTS = {'\\"o': "ö", '\\"u': "ü", '\\"a': "ä", "\\'e": "é", "\\'a": "á"}


def _plain(name: str) -> str:
    """BibTeX name -> text: decode the accents we use, drop grouping braces."""
    for k, v in _ACCENTS.items():
        name = name.replace(k, v)
    return name.replace("{", "").replace("}", "").strip()


def _bib() -> dict[str, tuple[str, str]]:
    """key -> (author label, year)."""
    text = (HERE / "references.bib").read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", text, re.S):
        key, body = m.group(1).strip(), m.group(2)
        a = re.search(r"author\s*=\s*\{(.*?)\},?\s*\n", body, re.S)
        y = re.search(r"year\s*=\s*\{?(\d{4})", body)
        authors = [s.strip() for s in a.group(1).replace("\n", " ").split(" and ")] if a else ["?"]
        last = [_plain(x.split(",")[0]) for x in authors]
        if "others" in last or len(last) > 2:
            label = f"{last[0]} et al."
        elif len(last) == 2:
            label = f"{last[0]} and {last[1]}"
        else:
            label = last[0]
        out[key] = (label, y.group(1) if y else "n.d.")
    return out


def _labels() -> dict[str, str]:
    aux = (HERE / "main.aux").read_text(encoding="utf-8")
    return {k: v for k, v in re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", aux)}


def _macro(src: str, name: str, render) -> str:
    """Replace \\name{arg} with render(arg), matching braces (arg may nest)."""
    out, i, tag = [], 0, "\\" + name + "{"
    while (j := src.find(tag, i)) != -1:
        k, depth = j + len(tag), 1
        while depth:
            depth += {"{": 1, "}": -1}.get(src[k], 0)
            k += 1
        out += [src[i:j], render(src[j + len(tag):k - 1])]
        i = k
    return "".join(out) + src[i:]


def main() -> None:
    bib, labels = _bib(), _labels()
    tex = (HERE / "main.tex").read_text(encoding="utf-8")
    title = re.search(r"\\title\{(.*?)\}", tex).group(1)
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1)
    body = re.search(r"\\maketitle(.*?)\\bibliography", tex, re.S).group(1)
    body = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", "", body, flags=re.S)
    src = f"\\section*{{Abstract}}\n{abstract}\n{body}"

    def cite(m, textual=False):
        parts = [bib.get(k.strip(), (k.strip(), "?")) for k in m.group(1).split(",")]
        if textual:
            return "; ".join(f"{a} ({y})" for a, y in parts)
        return "(" + "; ".join(f"{a} {y}" for a, y in parts) + ")"

    src = re.sub(r"\\citet\{([^}]*)\}", lambda m: cite(m, True), src)
    src = re.sub(r"\\cite[p]?\{([^}]*)\}", cite, src)
    src = re.sub(r"~?\\ref\{([^}]*)\}", lambda m: " " + labels.get(m.group(1), "?"), src)
    src = _macro(src, "TBD", lambda arg: f"\\textbf{{TBD: {arg}}}")
    src = re.sub(r"\\includegraphics(\[[^]]*\])?\{([^}]*)\}",
                 r"\\emph{[figure: latex/\2]}", src)

    md = subprocess.run(["pandoc", "-f", "latex", "-t", "gfm", "--wrap=none", "--shift-heading-level-by=1"],
                        input=src, capture_output=True, text=True, timeout=120)
    if md.returncode:
        raise SystemExit(f"pandoc failed: {md.stderr}")
    md = md.stdout
    authors = "An Vuong, Anthony Q. Nguyen, Thuan Nguyen, Thinh Nguyen"
    header = (f"# {title}\n\n{authors}\n\n"
              "> Generated from `docs/latex/main.tex` by `docs/latex/tex2md.py`; "
              "edit the LaTeX, not this file.\n\n")
    OUT.write_text(header + md, encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
