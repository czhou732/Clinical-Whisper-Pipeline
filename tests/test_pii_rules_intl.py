"""Safety-net rules for Chinese, Japanese, Korean and Hindi, and the second name check."""

import name_sweep
import pii_rules


def _numberer():
    ids = {}

    def number(label, key):
        n = ids.get((label, key)) or 1 + sum(1 for (lab, _) in ids if lab == label)
        ids[(label, key)] = n
        return n
    return number


def run(text, lang):
    return pii_rules.apply(text, _numberer(), lang=lang)[0]


def test_chinese_relation_title_and_city():
    assert run("我姐姐杨帆住在杭州。", "zh") == "我姐姐[full_name_1]住在[city_1]。"
    assert "王" not in run("王丽医生很好。", "zh")


def test_chinese_half_masked_name_is_finished():
    assert run("我和[last_name_1]洋是朋友。", "zh") == "我和[last_name_1][first_name_1]是朋友。"
    assert run("我姐姐黄[first_name_1]住在这里。", "zh").startswith("我姐姐[last_name_1][first_name_1]")


def test_chinese_ordinary_sentence_untouched():
    text = "我最近睡得不好，每天都很累。"
    assert run(text, "zh") == text


def test_korean_particles_stay_outside_the_tag():
    assert run("제 언니 정수빈은 부산에 살아요.", "ko") == "제 언니 [full_name_1]은 [city_1]에 살아요."
    assert run("담당 의사는 최현우 선생님이에요.", "ko") == "담당 의사는 [full_name_1] 선생님이에요."


def test_korean_city_needs_a_word_edge():
    # 광주리 is a basket, not Gwangju.
    assert run("광주리를 샀어요.", "ko") == "광주리를 샀어요."


def test_japanese_relation_and_title():
    assert run("姉の伊藤さくらが毎日電話をくれます。", "ja") == "姉の[full_name_1]が毎日電話をくれます。"
    assert "中村" not in run("主治医は中村先生です。", "ja")


def test_hindi_surname_relation_and_city():
    out = run("मेरी बहन प्रिया शर्मा पुणे में रहती है।", "hi")
    assert "प्रिया" not in out and "शर्मा" not in out and "पुणे" not in out
    assert "में रहती है" in out


def test_english_text_unchanged_by_language_rules():
    assert run("I moved here last year.", "zh") == "I moved here last year."


def test_propagation_for_scripts_without_capitals():
    texts, n = pii_rules.propagate(["[full_name_1]说好。", "后来杨帆又来了。", "김민지는 좋아요."],
                                   [("杨帆", "[full_name_1]"), ("김민지", "[full_name_2]")])
    assert texts == ["[full_name_1]说好。", "后来[full_name_1]又来了。", "[full_name_2]는 좋아요."]
    assert n == 2


def test_second_check_masks_only_verbatim_names():
    asked = []

    def ask(prompt):
        asked.append(prompt)
        return "1. 孙悦\n2. Beijing\n- 常德\nNONE\nmi"  # "Beijing" is not in the text

    texts, n = name_sweep.mask(["上个星期孙悦来看我了。", "我们是在常德长大的。"],
                               ["上个星期孙悦来看我了。", "我们是在常德长大的。"], ask, _numberer())
    assert texts == ["上个星期[full_name_1]来看我了。", "我们是在[full_name_2]长大的。"]
    assert n == 2 and len(asked) == 1


def test_second_check_skips_lowercase_latin_and_already_masked():
    def ask(prompt):
        return "de\nRaúl\nElena"

    texts, n = name_sweep.mask(["Soy de [first_name_1].", "Raúl vino."], ["Soy de Elena.", "Raúl vino."],
                               ask, _numberer())
    assert texts[0] == "Soy de [first_name_1]."
    assert texts[1] == "[full_name_2] vino." or texts[1].endswith(" vino.") and "Raúl" not in texts[1]


def test_second_check_ignores_text_after_the_end_marker():
    def ask(prompt):
        return "Here are the names:\n\n1. 常德<|eot_id|><|start_header_id|>assistant\n1. 我们"

    texts, n = name_sweep.mask(["我们是在常德长大的。"], ["我们是在常德长大的。"], ask, _numberer())
    assert texts == ["我们是在[full_name_1]长大的。"] and n == 1


def test_second_check_finishes_a_partly_masked_name():
    def ask(prompt):
        return "山田拓海\n马超"

    texts, _ = name_sweep.mask(["[last_name_1]拓[first_name_1]さんと", "我妈妈叫[first_name_1]超。", "超市很远。"],
                               ["山田拓海さんと", "我妈妈叫马超。", "超市很远。"], ask, _numberer())
    assert "拓" not in texts[0] and "超" not in texts[1]
    assert texts[2] == "超市很远。"  # a piece not touching a tag stays


def test_second_check_numbers_only_names_still_exposed():
    ids = {}

    def number(label, key):
        n = ids.get((label, key)) or 1 + sum(1 for (lab, _) in ids if lab == label)
        ids[(label, key)] = n
        return n

    texts, n = name_sweep.mask(["Vino [first_name_1] y Raúl."], ["Vino Elena y Raúl."],
                               lambda _: "Elena\nRaúl", number)
    assert n == 1 and len(ids) == 1
