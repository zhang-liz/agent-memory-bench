"""Synthetic on-call world: a seeded company whose facts arrive as messy messages.

The generator knows the true state at every step, so every question has an exact
answer. Facts are only ever revealed through free text (Slack, tickets, email),
never as clean records, so any structured memory has to extract them first.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

SERVICE_STEMS = [
    "payments", "ledger", "checkout", "inventory", "search", "auth", "billing",
    "notifications", "catalog", "pricing", "shipping", "returns", "fraud",
    "profile", "session", "media", "thumbnail", "recommendations", "reviews",
    "loyalty", "tax", "invoice", "refunds", "wallet", "geo", "routing",
    "dispatch", "telemetry", "metrics", "audit", "consent", "export", "import",
    "scheduler", "webhooks", "gateway", "cache", "feature-flags", "experiments",
    "reports", "messaging", "chat", "presence", "upload", "transcode", "cdn",
    "dns", "secrets", "identity", "partners", "quotes", "orders", "carts",
    "coupons", "subscriptions", "entitlements", "ratings", "maps", "eta",
    "settlement", "payouts", "kyc", "ledger-sync",
]
SERVICE_SUFFIXES = ["api", "svc", "db", "worker", "gateway", "queue"]
# A big world (scale > 1) needs more distinct services than there are stems, so
# stems get a domain prefix. Each prefixed stem is used once, so shorthand like
# "the eu-payments service" still names exactly one service.
BIG_PREFIXES = ["", "eu-", "us-", "ap-", "core-", "edge-", "int-", "legacy-"]
TEAM_NAMES = [
    "Falcon", "Otter", "Juniper", "Cobalt", "Maple", "Harbor", "Quartz",
    "Cedar", "Lynx", "Saffron", "Tundra", "Willow", "Basalt", "Heron",
    "Marigold", "Onyx",
]
PEOPLE = [
    "Priya", "Mateo", "Aiko", "Kwame", "Lena", "Rafael", "Sun-hee", "Tomasz",
    "Amara", "Diego", "Ingrid", "Farah", "Jonas", "Mei", "Olu", "Sofia",
    "Arjun", "Nadia", "Kenji", "Zara", "Emeka", "Lucia", "Ravi", "Hana",
    "Bilal", "Freya", "Tariq", "Yuki", "Chidi", "Elena", "Omar", "Signe",
    "Wen", "Lars", "Imani", "Pablo", "Noor", "Henrik", "Ama", "Kai",
    "Leila", "Bruno", "Asha", "Dmitri", "Rosa", "Femi", "Ines", "Taro",
]
SYMPTOMS = [
    "p99 latency spike", "elevated 5xx rate", "connection pool exhaustion",
    "memory leak after deploy", "stale cache reads", "TLS handshake failures",
    "consumer lag on the queue", "disk filling up", "deadlocks under load",
    "timeouts to upstream", "duplicate events", "clock skew errors",
]
CHATTER = [
    "anyone else seeing the coffee machine on 4 is broken again?",
    "reminder: all-hands moved to Thursday",
    "please fill in the offsite survey by Friday",
    "the staging cluster will be recycled tonight, expect blips",
    "lunch and learn on postgres vacuuming at noon",
    "new laptop images are rolling out this week",
    "who has the conference room booked at 3?",
    "the VPN client update is mandatory starting Monday",
]
LOG_LEVELS = ["INFO", "INFO", "INFO", "WARN", "DEBUG", "ERROR"]
LOG_MSGS = [
    "request completed status=200 latency_ms={n}",
    "cache miss key=user:{n}",
    "retrying upstream call attempt={k}",
    "gc pause ms={k}",
    "pool stats active={k} idle={k2}",
    "health check ok",
    "published event seq={n}",
    "slow query duration_ms={n}",
]

# Functional predicates hold one value per subject at a time.
FUNCTIONAL = {"owned_by", "on_call"}


@dataclass
class Event:
    step: int
    time: datetime
    kind: str
    text: str
    asserts: list[tuple[str, str, str]] = field(default_factory=list)
    retracts: list[tuple[str, str, str]] = field(default_factory=list)


@dataclass
class Question:
    qid: str
    after_step: int
    category: str
    text: str
    answer: list[str]


@dataclass
class World:
    seed: int
    n_steps: int
    events: list[Event]
    questions: list[Question]
    aliases: dict[str, str]  # lowercase alias -> canonical name
    final_state: dict | None = None  # true state after the last step

    def inbox(self) -> list[Event | Question]:
        """Events in order, with each question placed right after its step."""
        items: list[Event | Question] = []
        by_step: dict[int, list[Question]] = {}
        for q in self.questions:
            by_step.setdefault(q.after_step, []).append(q)
        for e in self.events:
            items.append(e)
            items.extend(by_step.get(e.step, []))
        return items


class _State:
    def __init__(self) -> None:
        self.owner: dict[str, str] = {}  # service -> team
        self.deps: dict[str, set[str]] = {}  # service -> services it calls
        self.members: dict[str, str] = {}  # person -> team
        self.oncall: dict[str, str] = {}  # team -> person
        self.incidents: list[tuple[str, str, str, int]] = []  # (id, service, symptom, step)
        self.changed_owner: set[str] = set()
        self.changed_oncall: set[str] = set()

    def dependents(self, svc: str) -> set[str]:
        """Every service that calls svc directly or through a chain."""
        out: set[str] = set()
        frontier = [svc]
        while frontier:
            cur = frontier.pop()
            for s, ds in self.deps.items():
                if cur in ds and s not in out:
                    out.add(s)
                    frontier.append(s)
        return out


def _team(t: str) -> str:
    return f"Team {t}"


def generate(seed: int, n_steps: int, scale: int = 1) -> World:
    """scale > 1 builds a company with 50 * scale services whose catalog arrives in
    batches, so the state outgrows what one note can carry. scale 1 is unchanged."""
    rng = random.Random(seed * 7919 + n_steps)
    big = scale > 1
    # The company stops growing at the 200-step size. Longer tasks get more
    # change instead: more handoffs, transfers and incidents per entity.
    size = min(n_steps, 200)
    n_services = max(8, size // 4)
    n_teams = max(3, size // 16)
    if big:
        n_services, n_teams = 50 * scale, len(TEAM_NAMES)
        if n_services > len(BIG_PREFIXES) * len(SERVICE_STEMS):
            raise ValueError(f"scale {scale} needs more service names than exist")
    n_people = n_teams * 3

    if big:
        combos = [p + s for p in BIG_PREFIXES for s in SERVICE_STEMS]
        stems = rng.sample(combos, n_services)
    else:
        stems = rng.sample(SERVICE_STEMS, n_services)
    services = [f"{s}-{rng.choice(SERVICE_SUFFIXES)}" for s in stems]
    teams = [_team(t) for t in rng.sample(TEAM_NAMES, n_teams)]
    people = rng.sample(PEOPLE, n_people)

    aliases: dict[str, str] = {}
    for svc, stem in zip(services, stems):
        aliases[svc.lower()] = svc
        aliases[stem] = svc
        aliases[f"the {stem} service"] = svc
    for t in teams:
        short = t.removeprefix("Team ")
        aliases[t.lower()] = t
        aliases[short.lower()] = t
    for p in people:
        aliases[p.lower()] = p

    st = _State()
    start = datetime(2026, 9, 1, 9, 0)
    events: list[Event] = []
    unintroduced = list(services)
    rng.shuffle(unintroduced)
    unstaffed = list(people)
    rng.shuffle(unstaffed)
    teams_without_oncall = list(teams)
    ticket_no = 4000

    def stem(svc: str) -> str:
        return svc.rsplit("-", 1)[0]

    def svc_ref(svc: str) -> str:
        """Services are often named by a shorthand in chat."""
        return rng.choice([svc, svc, stem(svc), f"the {stem(svc)} service"])

    def team_ref(t: str) -> str:
        short = t.removeprefix("Team ")
        return rng.choice([t, short, f"the {short} folks", f"{short} team"])

    intro_cutoff = int(n_steps * 0.6)
    snaps: dict[int, dict] = {}

    for step in range(1, n_steps + 1):
        now = start + timedelta(hours=step * 3 + rng.randint(0, 2), minutes=rng.randint(0, 59))
        need_intro = unintroduced or unstaffed or teams_without_oncall
        remaining_intro_steps = max(1, intro_cutoff - step)
        pressure = (len(unintroduced) + len(unstaffed) / 2 + len(teams_without_oncall)) / remaining_intro_steps
        if need_intro and (step >= intro_cutoff or rng.random() < min(0.95, 0.35 + pressure)):
            kind = _pick_intro(rng, unintroduced, unstaffed, teams_without_oncall, st)
        else:
            kind = _pick_later(rng, st, services)

        ev = Event(step=step, time=now, kind=kind, text="")
        author = rng.choice(people)

        if kind == "intro_service" and big:
            batch = [unintroduced.pop() for _ in range(min(len(unintroduced), rng.randint(6, 10)))]
            ticket_no += rng.randint(1, 9)
            rows = []
            for svc in batch:
                team = rng.choice(teams)
                known = [s for s in st.owner if s != svc]
                deps = set(rng.sample(known, k=min(len(known), rng.choice([0, 1, 1, 2, 2, 3]))))
                st.owner[svc] = team
                st.deps[svc] = deps
                ev.asserts.append((svc, "owned_by", team))
                ev.asserts.extend((svc, "depends_on", d) for d in sorted(deps))
                calls = ", ".join(svc_ref(d) for d in sorted(deps)) if deps else "nothing upstream"
                rows.append(f"- {svc}: owned by {team_ref(team)}; calls {calls}")
            ev.text = (f"TICKET OPS-{ticket_no} | Service catalog import from {author}\n"
                       f"Adding {len(batch)} services to the catalog:\n" + "\n".join(rows))
        elif kind == "intro_service":
            svc = unintroduced.pop()
            team = rng.choice(teams)
            known = [s for s in st.owner if s != svc]
            deps = set(rng.sample(known, k=min(len(known), rng.choice([0, 1, 1, 2, 2, 3]))))
            st.owner[svc] = team
            st.deps[svc] = deps
            ev.asserts.append((svc, "owned_by", team))
            ev.asserts.extend((svc, "depends_on", d) for d in sorted(deps))
            ticket_no += rng.randint(1, 9)
            ev.text = _render_intro_service(rng, svc, team, sorted(deps), author, ticket_no, svc_ref, team_ref)
        elif kind == "intro_person":
            person = unstaffed.pop()
            team = rng.choice(teams)
            st.members[person] = team
            ev.asserts.append((person, "member_of", team))
            ev.text = rng.choice([
                f"[#general] {author}: please welcome {person}, who joins {team_ref(team)} this week! 🎉",
                f"From: people-ops@acme.example\nSubject: New starter\n\n{person} starts Monday on {team_ref(team)}. Please help them get set up with prod access.",
                f"[#{_slug(team)}] {author}: {person} is moving over to us from another group, officially part of {team} as of today",
            ])
        elif kind == "intro_oncall":
            team = teams_without_oncall.pop()
            pool = [p for p, t in st.members.items() if t == team] or [rng.choice(people)]
            person = rng.choice(pool)
            st.oncall[team] = person
            ev.asserts.append((team, "on_call", person))
            ev.text = rng.choice([
                f"[#oncall] PagerDuty: {person} is now primary on-call for {team}.",
                f"[#{_slug(team)}] {author}: rotation set up, {person} has the pager for {team_ref(team)} this cycle",
                f"From: oncall-bot\nSubject: Rotation update\n\nPrimary on-call for {team}: {person}",
            ])
        elif kind == "transfer_owner":
            svc = rng.choice([s for s in st.owner])
            old = st.owner[svc]
            new = rng.choice([t for t in teams if t != old])
            st.owner[svc] = new
            st.changed_owner.add(svc)
            ev.retracts.append((svc, "owned_by", old))
            ev.asserts.append((svc, "owned_by", new))
            ev.text = rng.choice([
                f"[#platform-eng] {author}: heads up, {team_ref(new)} is taking over {svc_ref(svc)} from {team_ref(old)} starting today. ping them for anything {stem(svc)} related",
                f"From: eng-leads@acme.example\nSubject: Ownership change\n\nAfter the reorg, {svc} moves from {old} to {new}. Runbooks and alerts will be re-pointed this week.",
                f"[#{_slug(old)}] {author}: we've handed {svc_ref(svc)} off to {team_ref(new)}, it's theirs now",
            ])
        elif kind == "rotate_oncall":
            team = rng.choice(list(st.oncall))
            old = st.oncall[team]
            pool = [p for p, t in st.members.items() if t == team and p != old] or [p for p in people if p != old]
            new = rng.choice(pool)
            st.oncall[team] = new
            st.changed_oncall.add(team)
            ev.retracts.append((team, "on_call", old))
            ev.asserts.append((team, "on_call", new))
            ev.text = rng.choice([
                f"[#oncall] PagerDuty: on-call handoff for {team}. {old} → {new}.",
                f"[#{_slug(team)}] {old}: handing the pager to {new}, ping them for anything urgent",
                f"[#oncall] {author}: {new} is covering primary for {team_ref(team)} now, {old} is off rotation",
            ])
        elif kind == "add_dep":
            svc, dep = _new_edge(rng, st)
            st.deps[svc].add(dep)
            ev.asserts.append((svc, "depends_on", dep))
            ev.text = rng.choice([
                f"[#arch] {author}: FYI {svc_ref(svc)} now calls {svc_ref(dep)} for lookups, merged this morning",
                f"Design doc update: {svc} will take a hard dependency on {dep} starting with release 3.{rng.randint(1, 40)}.",
                f"[#deploys] {author}: shipped the change where {svc_ref(svc)} reads from {svc_ref(dep)}",
            ])
        elif kind == "drop_dep":
            svc = rng.choice([s for s, ds in st.deps.items() if ds])
            dep = rng.choice(sorted(st.deps[svc]))
            st.deps[svc].discard(dep)
            ev.retracts.append((svc, "depends_on", dep))
            ev.text = rng.choice([
                f"[#arch] {author}: {svc_ref(svc)} no longer talks to {svc_ref(dep)}, we removed that call path",
                f"[#deploys] {author}: decommissioned the {stem(dep)} integration in {svc}, it no longer depends on {dep}",
            ])
        elif kind == "incident":
            svc = rng.choice(list(st.owner))
            symptom = rng.choice(SYMPTOMS)
            inc = f"INC-{rng.randint(1000, 9999)}"
            st.incidents.append((inc, svc, symptom, step))
            ev.text = (
                f"INCIDENT {inc} opened by {author}\nService: {svc}\nSummary: {symptom}\n"
                f"Status: investigating\nNotes: first noticed via alerting, customer impact unclear."
            )
        else:  # chatter
            ev.text = f"[#general] {author}: {rng.choice(CHATTER)}"

        ev.text = ev.text + "\n\n" + _log_padding(rng, services, now)
        events.append(ev)
        snaps[step] = _snapshot(st)

    questions = _make_questions(rng, n_steps, snaps)
    return World(seed=seed, n_steps=n_steps, events=events, questions=questions, aliases=aliases,
                 final_state=snaps[max(snaps)])


def _snapshot(st: _State) -> dict:
    return {
        "owner": dict(st.owner),
        "deps": {k: set(v) for k, v in st.deps.items()},
        "members": dict(st.members),
        "oncall": dict(st.oncall),
        "incidents": list(st.incidents),
        "changed_owner": set(st.changed_owner),
        "changed_oncall": set(st.changed_oncall),
    }


def _pick_intro(rng, unintroduced, unstaffed, teams_without_oncall, st) -> str:
    options = []
    if unintroduced:
        options += ["intro_service"] * 3
    if unstaffed:
        options += ["intro_person"] * 2
    # An on-call assignment needs at least one known member to be realistic.
    if teams_without_oncall and st.members:
        options += ["intro_oncall"]
    return rng.choice(options or ["intro_service" if unintroduced else "intro_person"])


def _pick_later(rng, st: _State, services) -> str:
    options = ["chatter"] * 2
    if st.owner:  # always true at 200 steps and below, so those worlds are unchanged
        options = ["incident"] * 3 + options
    if len(st.owner) >= 2:
        options += ["transfer_owner"] * 2
    if st.oncall:
        options += ["rotate_oncall"] * 2
    if len(st.owner) >= 3 and _new_edge_possible(st):
        options += ["add_dep"]
    if any(st.deps.values()):
        options += ["drop_dep"]
    return rng.choice(options)


def _creates_cycle(st: _State, svc: str, dep: str) -> bool:
    return svc == dep or svc in _reachable(st, dep)


def _reachable(st: _State, start: str) -> set[str]:
    seen: set[str] = set()
    frontier = [start]
    while frontier:
        cur = frontier.pop()
        for d in st.deps.get(cur, ()):
            if d not in seen:
                seen.add(d)
                frontier.append(d)
    return seen


_SAMPLE_ABOVE = 100  # past this many services, find a new edge by sampling instead of scanning every pair


def _sample_edge(rng, st: _State) -> tuple[str, str] | None:
    svcs = list(st.owner)
    for _ in range(200):
        s, d = rng.choice(svcs), rng.choice(svcs)
        if d not in st.deps[s] and not _creates_cycle(st, s, d):
            return s, d
    return None


def _new_edge_possible(st: _State) -> bool:
    if len(st.owner) > _SAMPLE_ABOVE:
        return True  # with this many services an acyclic new edge always exists
    svcs = list(st.owner)
    return any(d not in st.deps[s] and not _creates_cycle(st, s, d) for s in svcs for d in svcs)


def _new_edge(rng, st: _State) -> tuple[str, str]:
    if len(st.owner) > _SAMPLE_ABOVE:
        edge = _sample_edge(rng, st)
        if edge:
            return edge
    svcs = list(st.owner)
    cands = [(s, d) for s in svcs for d in svcs if d not in st.deps[s] and not _creates_cycle(st, s, d)]
    return rng.choice(cands)


def _slug(team: str) -> str:
    return team.removeprefix("Team ").lower()


def _render_intro_service(rng, svc, team, deps, author, ticket_no, svc_ref, team_ref) -> str:
    dep_text = ", ".join(deps) if deps else "none"
    choice = rng.randrange(3)
    if choice == 0:
        return (
            f"TICKET OPS-{ticket_no} | Service catalog entry: {svc}\n"
            f"Owning team: {team}\n"
            f"Upstream dependencies: {dep_text}\n"
            f"Tier: {rng.choice(['1', '2', '3'])}\nRunbook: wiki/runbooks/{svc}"
        )
    if choice == 1:
        calls = (" It calls " + " and ".join(svc_ref(d) for d in deps) + ".") if deps else ""
        return (
            f"[#launches] {author}: {svc} is live in prod 🚀 {team_ref(team)} owns it, "
            f"page them if it misbehaves.{calls}"
        )
    deps_line = ("Depends on: " + "; ".join(deps)) if deps else "No upstream service dependencies."
    return (
        f"From: {author.lower()}@acme.example\nSubject: Handover notes for {svc}\n\n"
        f"{team} will own {svc} going forward. {deps_line}\nThanks!"
    )


def _log_padding(rng, services, now) -> str:
    """Realistic noise. Agents read logs all day, and logs cost tokens."""
    lines = []
    t = now - timedelta(minutes=5)
    for _ in range(rng.randint(14, 24)):
        t += timedelta(seconds=rng.randint(1, 20))
        msg = rng.choice(LOG_MSGS).format(n=rng.randint(10, 99999), k=rng.randint(1, 40), k2=rng.randint(1, 40))
        lines.append(
            f"{t:%Y-%m-%dT%H:%M:%SZ} {rng.choice(LOG_LEVELS):5} {rng.choice(services)} "
            f"trace={rng.getrandbits(48):012x} {msg}"
        )
    return "attached logs:\n" + "\n".join(lines)


CATEGORIES = ("lookup", "multihop", "transitive", "changed", "verbatim")


def _make_questions(rng: random.Random, n_steps: int, snaps: dict[int, dict]) -> list[Question]:
    questions: list[Question] = []
    qn = 0
    checkpoints = sorted({round(n_steps * k / 5) for k in range(1, 6)})
    # One question per category per checkpoint. A category with nothing to ask
    # yet (no ownership change before the first checkpoint, say) carries over to
    # the next checkpoint, so each category ends up with the same count.
    carried: list[str] = []
    for step in checkpoints:
        s = snaps[step]
        todo, carried = carried + list(CATEGORIES), []
        seen: set[str] = set()
        for cat in todo:
            q = _question(rng, cat, s)
            if q is None or q[0] in seen:
                carried.append(cat)
                continue
            seen.add(q[0])
            qn += 1
            text, answer = q
            questions.append(Question(qid=f"Q{qn}", after_step=step, category=cat, text=text, answer=answer))
    return questions


def followup_questions(w: World, per_category: int = 4) -> list[Question]:
    """Questions for a new session that starts after the inbox is finished.

    They ask about the final state, and none of them appeared while the inbox
    was being read, so whatever the first session kept has to cover them. A
    separate random stream keeps the first session's questions unchanged.
    """
    rng = random.Random(w.seed * 104729 + w.n_steps)
    out: list[Question] = []
    seen = {q.text for q in w.questions}
    for cat in CATEGORIES:
        made = 0
        for _ in range(50):
            if made == per_category:
                break
            q = _question(rng, cat, w.final_state)
            if q is None or q[0] in seen:
                continue
            seen.add(q[0])
            made += 1
            out.append(Question(qid=f"N{len(out) + 1}", after_step=w.n_steps, category=cat, text=q[0], answer=q[1]))
    return out


def _question(rng: random.Random, cat: str, s: dict) -> tuple[str, list[str]] | None:
    owner, deps, oncall = s["owner"], s["deps"], s["oncall"]
    if not owner:
        return None
    if cat == "lookup":
        svc = rng.choice(sorted(owner))
        return f"Which team currently owns {svc}?", [owner[svc]]
    if cat == "multihop":
        cands = [
            svc for svc, ds in deps.items()
            if ds and all(owner.get(d) in oncall for d in ds)
        ]
        if not cands:
            return None
        svc = rng.choice(sorted(cands))
        answer = sorted({oncall[owner[d]] for d in deps[svc]})
        return (
            f"{svc} is down and the cause looks upstream. We want to page the current primary on-call "
            f"person for every team that owns one of {svc}'s direct dependencies. Who do we page?",
            answer,
        )
    if cat == "transitive":
        tmp = _State()
        tmp.deps = deps
        # In a big world, early services have hundreds of dependents; keep answers to a list a person could check.
        cap = 12 if len(owner) > _SAMPLE_ABOVE else len(owner)
        cands = [svc for svc in sorted(owner) if 2 <= len(tmp.dependents(svc)) <= cap]
        if not cands:
            return None
        svc = rng.choice(cands)
        return (
            f"If {svc} goes down, which services break? List every service that depends on {svc} "
            f"directly or through a chain of dependencies.",
            sorted(tmp.dependents(svc)),
        )
    if cat == "changed":
        opts = [("owner", x) for x in sorted(s["changed_owner"])] + [("oncall", x) for x in sorted(s["changed_oncall"])]
        if not opts:
            return None
        kind, x = rng.choice(opts)
        if kind == "owner":
            return f"Which team owns {x} right now?", [owner[x]]
        return f"Who is the primary on-call for {x} right now?", [oncall[x]]
    if cat == "verbatim":
        incs = s["incidents"]
        if not incs:
            return None
        inc, svc, symptom, _ = rng.choice(incs)
        return f"What was the incident ID for the {symptom} reported on {svc}?", [inc]
    raise ValueError(cat)
