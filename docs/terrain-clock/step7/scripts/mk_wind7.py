"""Build wind7.py from wind5.py: the step-4 wind check with the never-worse rule (step 7) applied in every
re-derivation, the face-by-face reading taken from the comparison where the shipped decision was put in its place,
the receipt kept, and a refusal count at the actual winds alone."""
from pathlib import Path

here = Path(__file__).resolve().parent
s = (here / "wind5.py").read_text(encoding="utf-8").replace("\r\n", "\n")


def sub(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)


sub('''"""Check the local-face clock's wind reading''',
    '''"""wind7.py (step 7): wind5.py with the never-worse rule.  Where a grid's local-face decision was replaced by
the shipped measured clock's (``gpuwm.terrain_clock_local.never_worse``), the faces checked are the replaced
reading's, every re-derivation (as read, per hour, envelope) applies the same rule against the same shipped decision,
and each hour also counts faces whose limit at the actual wind alone (not the larger of read and actual) is below the
step that ran.  The receipt the preflight wrote is kept in the output.

Check the local-face clock's wind reading''')

sub('''        a = rec["adaptation"]
        self.a = a
''', '''        a = rec["adaptation"]
        self.a = a
        nw = getattr(a, "never_worse", None)
        self.nw = nw
        # The face-by-face derivation as read (the replaced one where the rule applied).
        la = nw.local if (nw is not None and nw.applied) else a
        self.la = la
''')
sub('''        self.count = int(a.local.sound_steps) if a.local is not None else int(a.time_step_sound)''',
    '''        self.count = int(la.local.sound_steps) if la.local is not None else int(la.time_step_sound)''')
sub('''        g = a.local.governing if a.local is not None else None''',
    '''        g = la.local.governing if la.local is not None else None''')

sub('''        a = _derive_measured(self.gid, self.run, rec["dt"], rec["slope"], rec["crest"], table=self.table,
                             read=combined.__getitem__)
        return _decision(a)''', '''        a = _derive_measured(self.gid, self.run, rec["dt"], rec["slope"], rec["crest"], table=self.table,
                             read=combined.__getitem__)
        local_decision = {**_decision(a), "status": a.status}
        if self.nw is not None:
            from gpuwm.terrain_clock_local import never_worse
            a = never_worse(a, self.nw.shipped, self.run)
        out = {**_decision(a), "status": a.status, "local_face_decision": local_decision,
               "never_worse_applied": None if self.nw is None else bool(a.never_worse.applied),
               "never_worse_reason": None if self.nw is None else a.never_worse.reason}
        return out''')

sub('''        refuse = eff_h < self.run_per_km * (1.0 - 1e-9)
''', '''        refuse = eff_h < self.run_per_km * (1.0 - 1e-9)
        eff_a = self.limit_at(act)
        refuse_actual = eff_a < self.run_per_km * (1.0 - 1e-9)
''')
sub('''                         "would_refuse_step_run": int((refuse & sel).sum()),
''', '''                         "would_refuse_step_run": int((refuse & sel).sum()),
                         "would_refuse_at_actual_wind": int((refuse_actual & sel).sum()),
                         "lowest_limit_at_actual_only_s": round(float(eff_a[sel].min()) * self.km, 3),
''')

sub('''                "step_setting_rule": self.setting_rule,
''', '''                "step_setting_rule": self.setting_rule,
                "never_worse": None if self.nw is None else self.nw.receipt(),
''')

sub('''        result["domains"][str(gid)] = {"header": head, "times": times, "envelope": env}''',
    '''        result["domains"][str(gid)] = {"header": head, "times": times, "envelope": env}
    result["terrain_clock_receipt"] = clock''')

sub('''                              "refuse": c["all"].get("would_refuse_step_run"), "same": c["hindsight_same"]}),''',
    '''                              "refuse": c["all"].get("would_refuse_step_run"),
                              "refuse_actual": c["all"].get("would_refuse_at_actual_wind"),
                              "same": c["hindsight_same"]}),''')

(here / "wind7.py").write_text(s, encoding="utf-8", newline="\n")
print("ok")
