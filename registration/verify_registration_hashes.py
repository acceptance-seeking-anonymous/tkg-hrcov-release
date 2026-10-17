#!/usr/bin/env python3
"""verify_registration_hashes.py -- verify every registered rule/protocol file of Manuscript A (code-release copy).

Convention (write_prereg and the earlier rule writers): the registration hash is computed BEFORE the file is written,
    recorded_sha256 = sha256( SERIALISE(body) ),  body = the rule with its registered_utc field, without the "sha256" field,
and the file is then written in insertion order with the "sha256" field embedded, so sha256(file bytes) != recorded_sha256.
SERIALISE is one of
    indent1_sorted : json.dumps(body, indent=1, sort_keys=True, default=str)   (tois_v7_rule, G8 rule and deviation B, ablation,
                                                                               G6, forward-holdout, v7.6 and v7.8 rules, and the
                                                                               two v7.8 external-comparator protocol notes)
    oneline_sorted : json.dumps(body, sort_keys=True, default=str)             (C4 protocol; default separators, one line)
The four earliest rules (main test, tkgl-icews Stage A and Stage B, pre-submission rule) carry a UTC time stamp but no hash.
The two v7.8 protocol notes carry their time stamp in the field created_utc (manifest key time_field); it is part of the hashed body.

Anonymisation: in the anonymous bundle, host names, IP addresses and local paths inside these files are replaced by <HOST>,
<SCRATCH>, <HOME> (same redaction as the rest of the bundle), so a redacted copy cannot reproduce the hash by itself.
Modes
    python3 verify_registration_hashes.py                      bundle-only: integrity of every bundled copy (sha256 listed in
                                                               registration/registration_manifest.json), embedded hash fields,
                                                               cross-references, and, for each redacted copy, its JSON structure
                                                               and placeholders against registration/redaction_map.json (the
                                                               full hash of a redacted file is verified only with --originals)
    python3 verify_registration_hashes.py --originals DIR      full: also recompute every recorded hash from the unredacted
                                                               originals in DIR (flat directory of the original files) and
                                                               check that the bundled
                                                               copy equals the redaction of the original
Unredacted originals are available to the editor and reviewers on request; they contain internal server addresses and are
not published.
Exit code 0 iff every check that can run passes.
Release copy: identical to the checker of the manuscript bundle except for the default --bundle (the repository root, since
this file sits in registration/), the flat-only lookup of --originals and (release v3) the optional manifest key time_field
(default registered_utc) naming the file's time-stamp field.
Release v4 (Manuscript A v7.9) adds the seven fresh-window documents:
  * v79_fresh_window_rule, its amendment 1, the implementation note and its addendum 1 carry a registration-time hash
    (indent1_sorted, as above; the amendment records the rule's hash, the freeze manifest records all four);
  * the freeze manifest, the uninspected-window check and the verdict carry no registration-time hash; the manifest records
    content_hash_kind = "file_sha256" and content_hash = sha256 of the original file bytes (computed at release), which is
    recomputed from the original (--originals) or, for an unredacted copy, from the bundled copy;
  * their redacted copies follow redaction_rules = "v2" of redaction_map.json: account name -> anon, server addresses and
    server numbers -> serverA / serverB / serverC (labels listed per entry), keys renamed by these rules listed under
    renamed_keys; the checker verifies that the map lists exactly the changed fields and the renamed keys;
    the v2 rules finally replace analysis-platform identifiers (tool name, job-store path segment, account-store
    organisation / project ids, language-model ids) by neutral text ("the authors' analysis environment", ".jobs", "<id>",
    "<LLM model id>"); the map gives their counts per changed field (platform). The patterns of these rules would name the
    platform, so they are not published: --originals reads them from DIR/platform_rules.json (provided with the originals);
    bundle-only mode does not need them;
  * other_copies: the same documents shipped under fresh_window/registration/ must be byte-identical to the copies here.
Release v4 also redacts, with the v1 placeholder <HOST>, bare server numbers that releases 1-3 had left in host-context
sentences of four v1 files (hardening_decision_rule, a_presub_rule, v76_rule, v76_rule_registered). The host rules (server
address-octet -> label table, host-number contexts) would name the hosts, so they are not published either: --originals
reads them from DIR/label_rules.json (load_label_rules); bundle-only mode uses only the generic address and path rules.
"""
import argparse, glob, hashlib, json, os, re, sys

SERIALISE = {
    "indent1_sorted": lambda b: json.dumps(b, indent=1, sort_keys=True, default=str),
    "oneline_sorted": lambda b: json.dumps(b, sort_keys=True, default=str),
}
H = lambda b: hashlib.sha256(b).hexdigest()

# redaction applied to the anonymous bundle (literals split so that this file contains none of the redacted strings)
_SSH = "ss" + "h:"
_IP = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
_SCR = "cs_" + "scratch"
# host rules: the address-octet -> server-label table and the bare-host-number rules are not published (they would name the
# hosts); --originals reads them from DIR/label_rules.json, provided with the originals (load_label_rules). Without it, RULES
# and PRE_RULES contain only the generic address and path rules, which is all the bundle-only checks need.
V1_HOST_RULES = []   # v1 (and v2, through redact): a bare host number in a host context ("on <n>", "<n> GPU", "<n> RTX ...") -> <HOST>
RULES_HEAD = [
    (_SSH + r"\s*" + _IP, "<HOST>"), (_IP, "<HOST>"),
    ("/" + "home/" + r"[^/\s\"']+/" + _SCR, "<SCRATCH>"), (r"/?\b" + _SCR + r"\b", "<SCRATCH>"),
    ("/" + "Users/" + r"[^/\s\"']+", "<HOME>"), ("/" + "home/" + r"[^/\s\"']+", "<HOME>"),
]
RULES_TAIL = [(r"\b" + _SSH + r"[A-Za-z0-9_.-]+", "<HOST>")]
RULES = RULES_HEAD + RULES_TAIL
def redact(text):
    for rx, rep in RULES: text = re.sub(rx, rep, text)
    return text

# release v4: label-preserving pre-rules for the v7.9 documents (manifest key redaction_rules = "v2"). Addresses and bare host
# numbers become the neutral server labels used elsewhere in the release (by last address octet; no address prefix is written
# here), so that host-keyed objects keep distinct keys; the path rules above are applied afterwards.
SRV = {}
_Q = r"(?<![=\w.])(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?!\d|\.\d)"   # end: not followed by a digit (an address may be followed by "_")
PRE_RULES = []
LOG_LABEL_RULES = []   # logs and review documents: as PRE_RULES without the quoted-number and " / " rules (there such numbers are ids or counts)
LABELS = ("serverA", "serverB", "serverC")

def build_label_rules(srv=None, v1_host_rules=()):
    SRV.clear(); SRV.update(srv or {})
    V1_HOST_RULES[:] = [tuple(r) for r in v1_host_rules]
    RULES[:] = RULES_HEAD + V1_HOST_RULES + RULES_TAIL
    lab = lambda m, g=1: SRV[m.group(g)]
    rules = [(_SSH + r"\s*" + _Q, lambda m: SRV.get(m.group(4), "server")), (_Q, lambda m: SRV.get(m.group(4), "server")),
             (r"(?<![=\w.])\d{1,3}\.\d{1,3}\.\d{1,3}\.(?=[{<])", "server.")]
    log_skip = []
    if SRV:
        L = "(" + "|".join(SRV) + ")"
        rules += [
            (r"(?<=[\s(/,])\." + L + r"\b", lab),
            (r"(?<=# )" + L + r"(?=:)", lab),
            (r"(?<=the needed )" + L + r"(?= files)", lab),
            (r"(?<=[\"'])" + L + r"(?=[\"'])", lab),
            (r"\b" + L + r"(?= / )", lab),
            (r"(?<= / )" + L + r"\b", lab),
            (r"\b(on|ran on) " + L + r"\b", lambda m: f"{m.group(1)} {SRV[m.group(2)]}"),
            (r"(?<=serverA and |serverB and |serverC and )" + L + r"\b", lab),
            (r"\b" + L + r"(?= was scanned| scans?\b)", lab),
            (r"(?<=\bhost )" + L + r"\b", lab),
            (r"\b([a-z])" + L + r"\b", lambda m: f"{m.group(1)}_{SRV[m.group(2)]}"),   # host number inside a short name (cN -> c_serverA)
        ]
        log_skip = [rules[6][0], rules[7][0], rules[8][0]]
    PRE_RULES[:] = rules
    LOG_LABEL_RULES[:] = [r for r in rules if r[0] not in log_skip]
    return len(PRE_RULES)

def load_label_rules(path):
    d = json.load(open(path, encoding="utf-8"))
    return build_label_rules(d["servers"], [(r["pattern"], r["replacement"]) for r in d["v1_host_rules"]])

build_label_rules()

# release v4, applied last by the v2 rules: analysis-platform identifiers (tool name, job-store path segment, account-store
# organisation and project ids, language-model identifiers) -> neutral text
PLATFORM_RULES = []   # not published: loaded from platform_rules.json, which is provided with the unredacted originals

def load_platform_rules(path):
    PLATFORM_RULES[:] = [(r["pattern"], r["replacement"]) for r in json.load(open(path, encoding="utf-8"))["rules"]]
    return len(PLATFORM_RULES)
PLATFORM_TOKENS = ("the authors' analysis environment", ".jobs/", "<id>", "<LLM model id>")

def platform(text):
    for rx, rep in PLATFORM_RULES: text = re.sub(rx, rep, text)
    return text

def redact_v2(text):
    for rx, rep in PRE_RULES: text = re.sub(rx, rep, text)
    return platform(redact(text))

def redact_for(e):
    return redact_v2 if e.get("redaction_rules") == "v2" else redact

def rename_keys(o, f):  # original object with its keys passed through the redaction (values untouched)
    if isinstance(o, dict): return {f(k): rename_keys(v, f) for k, v in o.items()}
    if isinstance(o, list): return [rename_keys(v, f) for v in o]
    return o

def renamed_ptrs(o, f, p=""):  # JSON pointers (redacted space) of members whose key the redaction renamed
    out = []
    if isinstance(o, dict):
        for k, v in o.items():
            q = p + "/" + str(f(k)).replace("~", "~0").replace("/", "~1")
            if f(k) != k: out.append(q)
            out += renamed_ptrs(v, f, q)
    elif isinstance(o, list):
        for i, v in enumerate(o): out += renamed_ptrs(v, f, f"{p}/{i}")
    return out

def canonical_hash(obj, conv):
    body = {k: v for k, v in obj.items() if k != "sha256"}
    return H(SERIALISE[conv](body).encode())

def get_path(d, dotted):
    for k in dotted.split("."):
        d = d[int(k)] if isinstance(d, list) else d[k]
    return d

PLACEHOLDERS = ("<HOST>", "<SCRATCH>", "<HOME>")

def keypaths_all(o, p=""):  # JSON paths with value types, in file order (the structure fingerprint)
    if isinstance(o, dict):
        for k, v in o.items():
            q = f"{p}.{k}" if p else k; yield q, type(v).__name__; yield from keypaths_all(v, q)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            q = f"{p}[{i}]"; yield q, type(v).__name__; yield from keypaths_all(v, q)

def structure_sha(o):
    return H("\n".join(f"{a}\t{b}" for a, b in keypaths_all(o)).encode())

def leaves_ptr(o, p=""):  # leaf values keyed by RFC 6901 JSON Pointer
    if isinstance(o, dict):
        for k, v in o.items(): yield from leaves_ptr(v, p + "/" + str(k).replace("~", "~0").replace("/", "~1"))
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from leaves_ptr(v, f"{p}/{i}")
    else: yield p, o

def find_original(root, entry):
    flat = os.path.join(root, os.path.basename(entry["bundled_path"]))
    return flat if os.path.exists(flat) else None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--originals", default=None); ap.add_argument("--bundle", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), help="repository root (contains registration/)")
    ap.add_argument("--json", action="store_true", help="print one JSON line per file")
    a = ap.parse_args()
    if a.originals:
        pr = os.path.join(a.originals, "platform_rules.json")
        if not os.path.exists(pr):
            print(f"--originals: {pr} (provided with the originals) is required for the v2 redaction"); sys.exit(2)
        load_platform_rules(pr)
        lr = os.path.join(a.originals, "label_rules.json")
        if not os.path.exists(lr):
            print(f"--originals: {lr} (provided with the originals) is required for the host rules"); sys.exit(2)
        load_label_rules(lr)
    man = json.load(open(os.path.join(a.bundle, "registration", "registration_manifest.json"), encoding="utf-8"))
    rmap = {f["file"]: f for f in json.load(open(os.path.join(a.bundle, "registration", "redaction_map.json"), encoding="utf-8"))["files"]}
    if not a.json:
        print("mode: " + ("full (--originals): recorded hashes recomputed from the unredacted originals" if a.originals else
                          "bundle-only: redacted copies are checked for integrity, structure and placeholders; their hashes need --originals"))
    allok = True; rows = []
    for e in man["files"]:
        r = {"file": e["bundled_path"], "registered_utc": e["registered_utc"], "recorded_sha256": e["recorded_sha256"], "convention": e["convention"]}
        p = os.path.join(a.bundle, e["bundled_path"]); raw = open(p, "rb").read(); obj = json.loads(raw)
        r["bundle_copy_intact"] = H(raw) == e["bundled_file_sha256"]
        tf = e.get("time_field", "registered_utc")  # the v7.8 protocol notes are time-stamped in created_utc
        r["registered_utc_matches"] = obj.get(tf, e["registered_utc"]) == e["registered_utc"]
        rf = redact_for(e); v2 = e.get("redaction_rules") == "v2"
        if e.get("content_hash_kind") == "file_sha256" and not e["redacted"]:  # v7.9 outcome documents: no registration-time hash
            r["file_hash_matches"] = H(raw) == e["content_hash"]
        if e["recorded_sha256"] and e.get("recorded_in") == "embedded":
            r["embedded_hash_matches"] = obj.get("sha256") == e["recorded_sha256"]
        for c in e.get("cited_in", []):
            try: val = get_path(json.load(open(os.path.join(a.bundle, c["file"]), encoding="utf-8")), c["key"])
            except Exception: val = None
            r.setdefault("citations_match", []).append({"file": c["file"], "key": c["key"], "ok": val == e["recorded_sha256"]})
        for c in e.get("file_sha256_cited_in", []):  # citations of the sha256 of the original file bytes
            try: val = get_path(json.load(open(os.path.join(a.bundle, c["file"]), encoding="utf-8")), c["key"])
            except Exception: val = None
            r.setdefault("citations_match", []).append({"file": c["file"], "key": c["key"], "ok": val == e["original_file_sha256"]})
        for other in e.get("other_copies", []):
            r.setdefault("other_copies_identical", []).append({"file": other, "ok": open(os.path.join(a.bundle, other), "rb").read() == raw})
        if e.get("canonical_bytes_of"):  # a *_registered.json file: its bytes are the hashed serialisation
            src = json.loads(open(os.path.join(a.bundle, e["canonical_bytes_of"]), "rb").read())
            r["registered_bytes_equal_serialisation_of_bundled_rule"] = raw == SERIALISE[e["convention"]]({k: v for k, v in src.items() if k != "sha256"}).encode()
            r["registered_bytes_hash"] = (H(raw) == e["recorded_sha256"]) if not e["redacted"] else None
        if e["recorded_sha256"] and not e["redacted"] and not e.get("canonical_bytes_of"):
            r["hash_recomputed_from_bundle"] = canonical_hash(obj, e["convention"]) == e["recorded_sha256"]
        # structure and placeholders (bundle-only): the redaction changes string values only, at the fields listed in the map
        r["structure_matches_manifest"] = structure_sha(obj) == e["structure_sha256"]
        lp = dict(leaves_ptr(obj))
        ph_fields = sorted(k for k, v in lp.items() if isinstance(v, str) and any(ph in v for ph in PLACEHOLDERS))
        m = rmap.get(e["bundled_path"])
        if e["redacted"]:
            r["listed_in_redaction_map"] = m is not None
            if m is not None:
                r["map_structure_matches"] = m["structure_sha256"] == e["structure_sha256"]
                r["placeholder_fields_match_map"] = (ph_fields == sorted(c["json_pointer"] for c in m["changed_fields"])) if not v2 else \
                    (set(ph_fields) <= {c["json_pointer"] for c in m["changed_fields"]})
                if v2:
                    r["server_labels_match_map"] = all({lb: lp.get(c["json_pointer"], "").count(lb) for lb in LABELS if lb in str(lp.get(c["json_pointer"], ""))} == c.get("labels", {}) for c in m["changed_fields"])
                    r["platform_tokens_match_map"] = all({t: str(lp.get(c["json_pointer"], "")).count(t) for t in PLATFORM_TOKENS if t in str(lp.get(c["json_pointer"], ""))} == c.get("platform", {}) for c in m["changed_fields"] if "platform" in c)
                r["placeholder_counts_match_map"] = all({ph: lp.get(c["json_pointer"], "").count(ph) for ph in PLACEHOLDERS if ph in lp.get(c["json_pointer"], "")} == c["placeholders"] for c in m["changed_fields"])
        else:
            r["no_placeholders_in_unredacted_copy"] = not ph_fields and m is None
        r["no_unredacted_host_or_path_pattern"] = rf(raw.decode("utf-8")) == raw.decode("utf-8")
        if a.originals:
            op = find_original(a.originals, e)
            if op is None: r["original_found"] = False
            else:
                oraw = open(op, "rb").read(); r["original_found"] = True
                r["original_bytes_match_manifest"] = H(oraw) == e["original_file_sha256"]
                if e["recorded_sha256"]:
                    r["hash_recomputed_from_original"] = (H(oraw) == e["recorded_sha256"]) if e.get("canonical_bytes_of") else (canonical_hash(json.loads(oraw), e["convention"]) == e["recorded_sha256"])
                r["bundle_copy_is_redaction_of_original"] = rf(oraw.decode("utf-8")).encode("utf-8") == raw
                if e.get("content_hash_kind") == "file_sha256": r["file_hash_matches_original"] = H(oraw) == e["content_hash"]
                lo = dict(leaves_ptr(json.loads(oraw)))
                if v2:  # keys may be renamed by the label rules: compare in the redacted pointer space
                    ok_ = json.loads(oraw); lo = dict(leaves_ptr(rename_keys(ok_, rf)))
                    r["map_lists_exactly_the_changed_fields"] = sorted(k for k in lo if lo[k] != lp.get(k)) == (sorted(c["json_pointer"] for c in m["changed_fields"]) if m else [])
                    r["map_lists_exactly_the_renamed_keys"] = sorted(renamed_ptrs(ok_, rf)) == sorted((m or {}).get("renamed_keys", []))
                else:
                    r["map_lists_exactly_the_changed_fields"] = sorted(k for k in lo if lo[k] != lp.get(k)) == (sorted(c["json_pointer"] for c in m["changed_fields"]) if m else [])
        checks = [v for k, v in r.items() if isinstance(v, bool)] + [x["ok"] for k in ("citations_match", "other_copies_identical") for x in r.get(k, [])]
        r["ok"] = all(checks)
        fh = e.get("content_hash_kind") == "file_sha256"
        if not r["ok"]: r["status"] = "FAIL"
        elif fh:
            red = ("bundled copy = redaction of original" if e["redacted"] else "unredacted copy, identical to original") if a.originals else \
                  ("redacted copy; structure, placeholders and labels verified" if e["redacted"] else "unredacted copy")
            r["status"] = ("PASS: file hash recomputed from original (no registration-time hash); " if a.originals else
                           "PASS: file hash recomputed from bundled copy (no registration-time hash); " if not e["redacted"] else
                           "FILE HASH (no registration-time hash), recomputable with --originals; ") + red
        elif a.originals:
            red = "bundled copy = redaction of original" if e["redacted"] else "unredacted copy, identical to original"
            r["status"] = (f"PASS: hash recomputed from original; {red}" if e["recorded_sha256"] else f"NO RECORDED HASH (time stamp only); {red}")
        elif e["redacted"]:
            r["status"] = ("PASS: structure and placeholders verified (bundle-only); full hash only with --originals" if e["recorded_sha256"]
                           else "NO RECORDED HASH (time stamp only); structure and placeholders verified (bundle-only)")
        else:
            r["status"] = ("PASS: hash recomputed from bundled copy" if e["recorded_sha256"] else "NO RECORDED HASH (time stamp only); unredacted copy")
        allok &= r["ok"]; rows.append(r)
        print(json.dumps(r) if a.json else f"{r['status']:<78} {e['bundled_path']}")
    print("ALL CHECKS PASSED" if allok else "SOME CHECKS FAILED")
    sys.exit(0 if allok else 1)

if __name__ == "__main__":
    main()
