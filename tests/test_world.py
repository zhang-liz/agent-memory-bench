from collections import Counter

from amb import world
from amb.grade import grade_set


def test_generation_is_deterministic():
    a, b = world.generate(3, 100), world.generate(3, 100)
    assert [e.text for e in a.events] == [e.text for e in b.events]
    assert [(q.text, q.answer) for q in a.questions] == [(q.text, q.answer) for q in b.questions]


def test_each_category_gets_five_questions():
    for n in (50, 100, 200):
        for seed in range(1, 6):
            w = world.generate(seed, n)
            assert Counter(q.category for q in w.questions) == {c: 5 for c in world.CATEGORIES}


def test_questions_only_ask_about_revealed_facts():
    w = world.generate(2, 100)
    for q in w.questions:
        if q.category == "verbatim":
            assert any(q.answer[0] in e.text for e in w.events if e.step <= q.after_step)


def test_facts_reach_the_agent_only_as_text():
    w = world.generate(1, 50)
    for e in w.events:
        for s, p, o in e.asserts:
            body = e.text.lower()
            # Each fact's subject is named in the message, possibly by a short form.
            assert s.lower().rsplit("-", 1)[0].removeprefix("team ") in body


def test_grader_accepts_aliases_and_rejects_extras():
    w = world.generate(1, 50)
    team = next(q for q in w.questions if q.category == "lookup").answer[0]
    short = team.removeprefix("Team ")
    assert grade_set([short], [team], w.aliases)["correct"]
    assert grade_set([f"the {short} folks"], [team], w.aliases)["correct"]
    assert not grade_set([team, "Nobody"], [team], w.aliases)["correct"]
    assert grade_set([team, "Nobody"], [team], w.aliases)["f1"] == 2 / 3
