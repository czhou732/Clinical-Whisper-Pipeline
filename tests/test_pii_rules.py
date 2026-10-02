"""Safety-net rules run after OpenMED."""

import pii_rules


def run(text):
    ids = {}

    def number(label, key):
        n = ids.get((label, key)) or 1 + sum(1 for (lab, _) in ids if lab == label)
        ids[(label, key)] = n
        return n

    return pii_rules.apply(text, number)[0]


def test_relationship_names():
    assert run("I live with my sister Elena, and her kids.") == "I live with my sister [first_name_1], and her kids."
    assert run("my friend And then we left") == "my friend And then we left"


def test_titles_institutions_and_workplaces():
    assert run("Doctor Feldman said so.") == "Doctor [last_name_1] said so."
    assert run("at Saint [city_1]'s Hospital now") == "at [organization_1] now"
    assert run("I went to the Hollis Clinic.") == "I went to the [organization_1]."
    assert run("I work at Kroger now.") == "I work at [organization_1] now."
    assert run("The hospital was fine.") == "The hospital was fine."


def test_spoken_numbers_and_emails():
    assert run("reach me at five five five, one two three four") == "reach me at [phone_number_1]"
    assert run("at [time_1], two [time_2].") == "at [phone_number_1]."
    assert run("I have one or two friends.") == "I have one or two friends."
    assert run("It's lucia dot liu at gmail dot com.") == "It's [email_1]."
    assert run("It's lucia [user_name] [email], ok") == "It's [email_1], ok"
    assert run("My email is [email_1].") == "My email is [email_1]."


def test_same_name_same_number():
    assert run("my sister Elena called. My cousin Elena too.").count("[first_name_1]") == 2


def test_propagation_masks_later_mentions_only_of_caught_names():
    texts = ["I live with my sister [first_name_1].", "Elena called again.", "I will call. Will you?"]
    out, n = pii_rules.propagate(texts, [("Elena", "[first_name_1]")])
    assert out[1] == "[first_name_1] called again." and n == 1
    assert out[2] == texts[2]


def test_workplace_with_words_before_at():
    assert run("He works nights down at Kroger.") == "He works nights down at [organization_1]."
    assert run("I work for myself now.") == "I work for myself now."


def test_street_names_and_written_phone_numbers():
    assert run("We lived on Ocean Drive back then.") == "We lived on [street_address_1] back then."
    assert run("Call (213) 555-0187 or 555-6137.") == "Call [phone_number_1] or [phone_number_2]."
    assert run("In 2019-2020 it cost 300-400 dollars.") == "In 2019-2020 it cost 300-400 dollars."


def test_half_tagged_street_and_number_are_finished():
    assert run("We lived on Ocean [street_address_1] back then.") == "We lived on [street_address_1] back then."
    assert run("Her number's 555[phone_number_1], okay.") == "Her number's [phone_number_1], okay."
