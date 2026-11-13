"""A small grammar that writes many wordings of the four causal claims (training data for the 2B parser).

Why a grammar and not a hand-written list: with ~40 sentences DistilBERT memorises single words
("cause of" -> reversed, "some" -> confounder). Here every class uses the *same* verbs and the same
"third variable" phrases, so the only way to tell the classes apart is the sentence structure:
who is the subject, active or passive voice, "directly" vs "via another variable".

{x} is always the cause slot and {y} the effect slot (collider / confounder are symmetric).
Words of the official paraphrase test set ("affects", "influences", "mediator", "together",
"some ... variable(s)") are deliberately not used, so that test measures unseen vocabulary.
"""
from __future__ import annotations

from .parsing import COLLIDER, CONFOUNDER, DIRECT, INDIRECT

# (3rd person singular, plural, past participle)
VERBS = [
    ("causes", "cause", "caused"),
    ("produces", "produce", "produced"),
    ("drives", "drive", "driven"),
    ("brings about", "bring about", "brought about"),
    ("triggers", "trigger", "triggered"),
    ("controls", "control", "controlled"),
    ("determines", "determine", "determined"),
    ("generates", "generate", "generated"),
]
# Modifier before the verb ("directly causes") or after the object ("causes B with nothing in between").
MODIFIERS = {
    DIRECT: (["directly", "immediately"],
             ["with nothing in between", "without any intermediate variable", "in a single step"]),
    INDIRECT: (["indirectly"],
               ["via an intermediate variable", "through another variable", "by way of other variables",
                "only through a chain of other variables", "through at least one variable in between"]),
}
# Noun-phrase claims: (cause-subject form, effect-subject form)
NOUN_FORMS = {
    DIRECT: (["{x} is a direct cause of {y}.", "{x} is a parent of {y} in the causal graph.",
              "{x} is an immediate cause of {y}.", "There is a direct causal link from {x} to {y}.",
              "{x} has a direct causal effect on {y}."],
             ["{y} is a direct effect of {x}.", "{y} is a child of {x} in the causal graph.",
              "{y} is an immediate consequence of {x}.", "The direct cause of {y} is {x}.",
              "{y} depends directly on {x}."]),
    INDIRECT: (["{x} is an indirect cause of {y}.", "{x} is an ancestor of {y} but not its parent.",
                "{x} has an indirect causal effect on {y}.",
                "There is a causal chain from {x} to {y} with at least one variable in between.",
                "{x} causes a variable that in turn causes {y}."],
               ["{y} is an indirect effect of {x}.", "{y} is a descendant of {x} but not its child.",
                "{y} is an indirect consequence of {x}.", "The indirect cause of {y} is {x}.",
                "{y} depends on {x} only through other variables."]),
}
THIRD = ["another variable", "a third variable", "a shared variable", "one other variable", "a common variable"]
SHARED_NOUNS = {
    COLLIDER: ["common effect", "common consequence", "shared outcome", "common descendant", "joint effect"],
    CONFOUNDER: ["common cause", "common source", "shared origin", "common ancestor", "joint cause"],
}


def _cap(text: str) -> str:
    """Sentence-initial form. 'a third variable' -> 'The third variable': a capital 'A' would be read as a variable."""
    if text.startswith("a "):
        return "The " + text[2:]
    return text[0].upper() + text[1:]


def chain_templates(relation: str) -> list[str]:
    """direct / indirect: active and passive voice, modifier before or after."""
    before, after = MODIFIERS[relation]
    out = []
    for singular, _, participle in VERBS:
        out += [f"{{x}} {m} {singular} {{y}}." for m in before]
        out += [f"{{x}} {singular} {{y}} {m}." for m in after]
        out += [f"{{y}} is {m} {participle} by {{x}}." for m in before]
        out += [f"{{y}} is {participle} by {{x}} {m}." for m in after]
    active, passive = NOUN_FORMS[relation]
    return out + active + passive


def collider_templates() -> list[str]:
    """X and Y both cause a third variable (arrows point *into* the third variable)."""
    out = []
    for singular, plural, participle in VERBS:
        for third in THIRD:
            out += [f"{{x}} and {{y}} both {plural} {third}.",
                    f"{{x}} and {{y}} each {singular} {third}.",
                    f"{_cap(third)} is {participle} by both {{x}} and {{y}}.",
                    f"{_cap(third)} is {participle} by {{x}} and also by {{y}}."]
    for noun in SHARED_NOUNS[COLLIDER]:
        out += [f"{{x}} and {{y}} share a {noun}.", f"{{x}} and {{y}} have at least one {noun}.",
                f"There is a {noun} of {{x}} and {{y}}."]
    return out


def confounder_templates() -> list[str]:
    """A third variable causes both X and Y (arrows point *out of* the third variable)."""
    out = []
    for singular, _, participle in VERBS:
        for third in THIRD:
            out += [f"{_cap(third)} {singular} both {{x}} and {{y}}.",
                    f"{_cap(third)} {singular} {{x}} and also {{y}}.",
                    f"{{x}} and {{y}} are both {participle} by {third}.",
                    f"{{x}} and {{y}} are each {participle} by {third}."]
    for noun in SHARED_NOUNS[CONFOUNDER]:
        out += [f"{{x}} and {{y}} share a {noun}.", f"{{x}} and {{y}} have at least one {noun}.",
                f"There is a {noun} of {{x}} and {{y}}."]
    return out


def all_templates() -> dict[str, list[str]]:
    return {DIRECT: chain_templates(DIRECT), INDIRECT: chain_templates(INDIRECT),
            COLLIDER: collider_templates(), CONFOUNDER: confounder_templates()}
