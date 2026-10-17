#!/usr/bin/env python3
"""verify_fresh_window.py -- check the released fresh-window package against its records.

The package was frozen and listed in FILES.md (SHA-256 per file) before the release; the frozen code files are also listed in the
freeze manifest (registration/fresh_window_freeze_manifest.json). For the anonymous release, the account name of the original
runs was replaced by 'anon', server addresses / bare host numbers by the labels serverA/serverB/serverC, and analysis-platform
identifiers (tool name, job-store path segment, account-store ids, language-model ids) by neutral text (fresh-window rules);
the registration documents use the registration checker's v2 rules. FILES_RELEASE.tsv gives, per file, the package SHA-256
(= FILES.md), the released SHA-256 and the redaction applied.

  python3 fresh_window/verify_fresh_window.py                   bundle-only: released files intact; unredacted files equal
                                                                 FILES.md; frozen files' package hashes equal the freeze manifest;
                                                                 registration copies equal ../registration/
  python3 fresh_window/verify_fresh_window.py --originals DIR   also: every original file (unredacted package, available to the
                                                                 editor and reviewers on request) equals FILES.md, and its redaction
                                                                 equals the released file byte for byte (DIR also holds
                                                                 platform_rules.json and label_rules.json, the unpublished platform and
                                                                 host rules; for the logs also account_rules.json)
  python3 fresh_window/verify_fresh_window.py --originals DIR --log-bundle LOGS --docs DOCS
                                                                 also: the release-v4 additions logs/ and docs/ against the
                                                                 delivered log bundle and review documents (ADDITIONS.tsv)
Exit code 0 iff every check passes.
"""
import argparse, csv, hashlib, importlib.util, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
H = lambda b: hashlib.sha256(b).hexdigest()
ADDED = {"README.md", "FILES_RELEASE.tsv", "ADDITIONS.tsv", "fw_env.sh", "fw_localize.sh", "verify_fresh_window.py", "provenance/README.md"}

# ---- fresh-window label rules: the same rules as ../registration/verify_registration_hashes.py (build_label_rules); the
# address-octet -> label table is not published and is read with --originals from DIR/label_rules.json
PRE_RULES = []
LOG_LABEL_RULES = []   # logs and review documents: PRE_RULES without the quoted-number and " / " rules (there such numbers are ids or counts)
def pre(text):
    n = 0
    for rx, rep in PRE_RULES:
        text, k = re.subn(rx, rep, text); n += k
    return text, n


# analysis-platform identifiers (same rules as ../registration/verify_registration_hashes.py), applied last
PLATFORM_RULES = []   # not published: loaded from <originals>/platform_rules.json (provided with the unredacted package)
def platform(text):
    n = 0
    for rx, rep in PLATFORM_RULES:
        text, k = re.subn(rx, rep, text); n += k
    return text, n

# release v4 log bundle (logs/, rebuilt as fresh_window_logs_v2) and review documents (docs/):
#   label rules (pre); account names of every server account other than ours -> user1, user2, ... (ACCOUNT_RULES: the names are
#   not published; read from account_rules.json, provided with the originals) and any remaining home directory of another
#   account -> /home/otheruser; process listings: ps lines of other accounts or of the system, and the GPU process rows of
#   nvidia-smi (which do not show the owner), are replaced by one marker line per block; platform rules last
ACCOUNT_RULES = []
OTHER_HOME = r"/home/(?!\[REDACTED|anon\b|otheruser\b|user\d+\b)[A-Za-z0-9_.-]+"
PS_HEAD = re.compile(r"^\s*PID\s+%CPU\s+ELAPSED\s+COMMAND\s*$")
PS_LINE = re.compile(r"^\s*\d+\s+[\d.]+\s+[\d:-]+\s")
GPU_ROW = re.compile(r"^\|\s+\d+\s+\S+\s+\S+\s+\d+\s+[CGM+]+\s+.*\|\s*$")
OWN = ("[REDACTED:~ANON]", "/home/anon/")

def load_account_rules(path):
    ACCOUNT_RULES[:] = [(r["pattern"], r["replacement"]) for r in json.load(open(path, encoding="utf-8"))["rules"]]
    return len(ACCOUNT_RULES)

def filter_processes(text):
    out = []; n = 0; L = text.split("\n"); i = 0
    while i < len(L):
        line = L[i]; out.append(line)
        if PS_HEAD.match(line):
            i += 1; k = 0
            while i < len(L) and PS_LINE.match(L[i]):
                if any(o in L[i] for o in OWN): out.append(L[i])
                else: k += 1
                i += 1
            if k: out.append(f"    [{k} process lines of other accounts or of the system removed for the release]"); n += k
            continue
        if GPU_ROW.match(line):
            out.pop(); k = 0
            while i < len(L) and GPU_ROW.match(L[i]): k += 1; i += 1
            out.append("|  [GPU process rows removed for the release: the table does not show their owner]".ljust(len(line) - 1) + "|"); n += k
            continue
        i += 1
    return "\n".join(out), n

# release wording (public, non-identifying): the shipped log bundle is v2 and lives in fresh_window/logs/; manuscript version
# labels are removed from the review documents and the bundle README
WORDING_RULES = [
    (r"`fresh_window_logs_v1\.tar\.gz`", "the log bundle v2 in `fresh_window/logs/`"),
    (r"fresh_window_logs_v1\.tar\.gz", "the log bundle v2 in fresh_window/logs/"),
    (r"fresh_window_logs_v1/", "fresh_window/logs/"),
    (r"(?m)^# fresh_window_logs_v1 -- logs supporting the v79 fresh-window chronology", "# fresh_window_logs_v2 -- logs supporting the fresh-window chronology"),
    (r"fresh_window_logs_v1", "fresh_window_logs_v2"),
    (r"(?m)^# v79 fresh-window chronology", "# Fresh-window chronology"),
    (r'"title": "v79 fresh-window chronology', '"title": "Fresh-window chronology'),
    (r"\bv79 fresh-window rule", "fresh-window rule"),
    (r"Manuscript v7\.9 \(Section 6\.4\) gives", "An earlier manuscript draft gave"),
]

def log_redact(text):
    n = {}; t = text; k = 0
    for rx, rep in LOG_LABEL_RULES:
        t, c = re.subn(rx, rep, t); k += c
    n["label"] = k
    k = 0
    for rx, rep in ACCOUNT_RULES:
        t, c = re.subn(rx, rep, t); k += c
    t, c = re.subn(OTHER_HOME, "/home/otheruser", t); n["account"] = k + c
    t, n["process_lines"] = filter_processes(t)
    t, n["platform"] = platform(t)
    k = 0
    for rx, rep in WORDING_RULES:
        t, c = re.subn(rx, rep, t); k += c
    n["wording"] = k
    return t, n

def fw_redact(text, user):
    t, _ = pre(text)
    t = re.sub(r"\b" + re.escape(user) + r"\b", "anon", t)
    return platform(t)[0]

def infer_user(texts):
    names = set()
    for t in texts: names.update(re.findall(r"/home/([A-Za-z0-9_.-]+)/" + "cs_" + "scratch", t))
    assert len(names) == 1, "cannot infer the account name from the originals"
    return names.pop()

def load_checker():
    spec = importlib.util.spec_from_file_location("vrh", os.path.join(ROOT, "registration", "verify_registration_hashes.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def files_md_table(path):
    t = open(path, encoding="utf-8").read()
    return {p: (h, int(n)) for p, h, n in re.findall(r"^\| `([^`]+)` \| `([0-9a-f]{64})` \| (\d+) \|", t, re.M)}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--originals", default=None, help="directory of the unredacted package (fresh_window_code_v1/)")
    ap.add_argument("--log-bundle", default=None, help="unpacked fresh_window_logs_v1/ as delivered (with --originals for the platform rules)")
    ap.add_argument("--docs", default=None, help="directory of the review documents as delivered (docs/)")
    a = ap.parse_args(); ok = True; n = 0
    rows = list(csv.DictReader(open(os.path.join(HERE, "FILES_RELEASE.tsv"), encoding="utf-8"), delimiter="\t"))
    fm = files_md_table(os.path.join(HERE, "FILES.md"))
    def chk(cond, msg):
        nonlocal ok
        if not cond: ok = False; print("FAIL", msg)
    listed = set()
    for r in rows:
        p = os.path.join(ROOT, r["release_path"]); listed.add(os.path.relpath(p, HERE)); raw = open(p, "rb").read(); n += 1
        chk(H(raw) == r["released_sha256"] and len(raw) == int(r["released_bytes"]), f"{r['release_path']}: released copy differs from FILES_RELEASE.tsv")
        if r["redaction"] == "none": chk(r["released_sha256"] == r["package_sha256"], f"{r['release_path']}: marked unredacted but differs from the package")
        if r["package_path"] in fm: chk(fm[r["package_path"]][0] == r["package_sha256"], f"{r['package_path']}: package hash differs from FILES.md")
        elif r["package_path"] != "FILES.md": chk(False, f"{r['package_path']}: not listed in FILES.md")
    chk(set(fm) == {r["package_path"] for r in rows if r["package_path"] != "FILES.md"}, "FILES.md and FILES_RELEASE.tsv list different files")
    # release-v4 additions (reproduce/, docs/, logs/): ADDITIONS.tsv gives the source and the released SHA-256 of every file
    adds = list(csv.DictReader(open(os.path.join(HERE, "ADDITIONS.tsv"), encoding="utf-8"), delimiter="\t"))
    lsums = {}
    if os.path.exists(os.path.join(HERE, "logs", "SHA256SUMS")):
        for line in open(os.path.join(HERE, "logs", "SHA256SUMS"), encoding="utf-8"):
            h, pth = line.rstrip("\n").split("  ", 1); lsums[pth.lstrip("./")] = h
    for r in adds:
        p = os.path.join(ROOT, r["release_path"]); listed.add(os.path.relpath(p, HERE)); raw = open(p, "rb").read()
        chk(H(raw) == r["released_sha256"] and len(raw) == int(r["released_bytes"]), f"{r['release_path']}: differs from ADDITIONS.tsv")
        if r["redaction"] == "none" and r["source_sha256"] and r["release_path"].startswith(("fresh_window/logs/", "fresh_window/docs/")):
            chk(r["released_sha256"] == r["source_sha256"], f"{r['release_path']}: marked unredacted but differs from its source")
        rl = os.path.relpath(p, os.path.join(HERE, "logs"))
        if r["release_path"].startswith("fresh_window/logs/") and rl != "SHA256SUMS":
            chk(lsums.get(rl) == r["released_sha256"], f"{r['release_path']}: differs from logs/SHA256SUMS (bundle v2)")
    chk(sorted(lsums) == sorted(os.path.relpath(os.path.join(ROOT, r["release_path"]), os.path.join(HERE, "logs")) for r in adds
                                 if r["release_path"].startswith("fresh_window/logs/") and not r["release_path"].endswith("logs/SHA256SUMS")),
        "logs/SHA256SUMS does not list exactly the released log files")
    present = {os.path.relpath(os.path.join(d, f), HERE) for d, _, fs in os.walk(HERE) for f in fs if "__pycache__" not in d}
    chk(present - listed - ADDED == set(), f"unlisted files: {sorted(present - listed - ADDED)}")
    # frozen code: package hash = freeze manifest (path suffix after the track directory)
    fz = json.load(open(os.path.join(HERE, "registration", "fresh_window_freeze_manifest.json"), encoding="utf-8"))
    fzh = {}
    for g in fz["groups"].values():
        for it in g:
            m = re.search(r"/phase2_fresh_window/((?:code|code_sa|code_ct)/.+)$", it.get("path", ""))
            if m: fzh[m.group(1)] = it["sha256"]
    nfz = 0
    for r in rows:
        if r["frozen"].startswith("yes"):
            nfz += 1; chk(fzh.get(r["package_path"]) == r["package_sha256"], f"{r['package_path']}: package hash differs from the freeze manifest")
    # registration copies = ../registration/ (same v2 redaction)
    for r in rows:
        if r["package_path"].startswith("registration/"):
            q = os.path.join(ROOT, "registration", os.path.basename(r["package_path"]))
            if os.path.exists(q): chk(open(q, "rb").read() == open(os.path.join(ROOT, r["release_path"]), "rb").read(), f"{r['release_path']}: differs from registration/ copy")
    print(f"bundle-only: {n} files, {nfz} frozen files checked against the freeze manifest; {len(adds)} release-v4 additions")
    if a.originals:
        V = load_checker(); texts = {}
        pr = os.path.join(a.originals, "platform_rules.json")
        if not os.path.exists(pr): print(f"--originals: {pr} (provided with the originals) is required"); sys.exit(2)
        V.load_platform_rules(pr); PLATFORM_RULES[:] = V.PLATFORM_RULES
        lr = os.path.join(a.originals, "label_rules.json")
        if not os.path.exists(lr): print(f"--originals: {lr} (provided with the originals) is required"); sys.exit(2)
        V.load_label_rules(lr); PRE_RULES[:] = V.PRE_RULES; LOG_LABEL_RULES[:] = V.LOG_LABEL_RULES
        for r in rows:
            op = os.path.join(a.originals, r["package_path"])
            if not os.path.exists(op): chk(False, f"original missing: {r['package_path']}"); continue
            texts[r["package_path"]] = open(op, "rb").read()
        user = infer_user([t.decode("utf-8") for t in texts.values()])
        for r in rows:
            raw = texts.get(r["package_path"])
            if raw is None: continue
            chk(H(raw) == r["package_sha256"], f"{r['package_path']}: original differs from FILES.md / FILES_RELEASE.tsv")
            red = V.redact_v2(raw.decode("utf-8")) if r["redaction"] == "registration_v2" else fw_redact(raw.decode("utf-8"), user)
            if r["redaction"] == "none": red = raw.decode("utf-8")
            chk(red.encode("utf-8") == open(os.path.join(ROOT, r["release_path"]), "rb").read(), f"{r['release_path']}: not the redaction of the original")
        print(f"originals: {len(texts)} files, each equal to FILES.md and its redaction equal to the released copy")
    for opt, prefix in ((a.log_bundle, "fresh_window/logs/"), (a.docs, "fresh_window/docs/")):
        if not opt: continue
        if not a.originals: chk(False, "--log-bundle / --docs need --originals (platform rules)"); continue
        ar = os.path.join(a.originals, "account_rules.json")
        if not os.path.exists(ar): chk(False, f"{ar} (provided with the originals) is required"); continue
        load_account_rules(ar)
        k = 0
        for r in adds:
            if not r["release_path"].startswith(prefix): continue
            op = os.path.join(opt, r["release_path"][len(prefix):])
            if not os.path.exists(op): chk(False, f"source missing: {op}"); continue
            raw = open(op, "rb").read(); k += 1; rel_out = open(os.path.join(ROOT, r["release_path"]), "rb").read()
            chk(H(raw) == r["source_sha256"], f"{r['release_path']}: source differs from ADDITIONS.tsv")
            if r["release_path"].endswith("logs/SHA256SUMS"): continue          # regenerated for the v2 bundle (checked above)
            try: red = log_redact(raw.decode("utf-8"))[0].encode("utf-8")
            except UnicodeDecodeError: red = raw
            if r["release_path"].endswith("logs/README.md"):                    # v2 README = redacted v1 README + release section
                chk(rel_out.startswith(red.rstrip(b"\n")), f"{r['release_path']}: does not start with the redacted v1 README"); continue
            chk(red == rel_out, f"{r['release_path']}: not the redaction of its source")
        print(f"{prefix}: {k} source files, each equal to ADDITIONS.tsv and its redaction equal to the released copy")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"); sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
