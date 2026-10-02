"""Telling which language a transcript is in."""

import pytest

from language import detect, is_english

SAMPLES = {
    "en": "So I think the thing is that I just don't really want to go out anymore, you know, "
          "and my sister keeps asking me what is wrong but I don't know what to tell her.",
    "es": "Pues yo creo que no tengo ganas de salir, mi hermana me pregunta qué me pasa pero "
          "no sé qué decirle, es muy difícil para mí y eso me pone triste.",
    "fr": "Je pense que je n'ai plus envie de sortir, ma sœur me demande ce qui ne va pas mais "
          "je ne sais pas quoi lui dire, c'est vraiment difficile pour moi.",
    "de": "Ich glaube, ich habe einfach keine Lust mehr rauszugehen, und meine Schwester fragt "
          "mich, was los ist, aber ich weiß nicht, was ich ihr sagen soll.",
    "zh": "我觉得我真的不想出去了，我姐姐一直问我怎么了，但是我不知道该怎么跟她说。",
    "ja": "もう外に出たくないと思っています。姉がどうしたのと聞いてきますが、何と言えばいいかわかりません。",
    "ko": "이제 정말 밖에 나가고 싶지 않아요. 언니가 무슨 일이냐고 계속 묻는데 뭐라고 해야 할지 모르겠어요.",
    "ar": "أعتقد أنني لا أريد الخروج بعد الآن، أختي تسألني ما الخطب لكنني لا أعرف ماذا أقول لها.",
}


@pytest.mark.parametrize("code", sorted(SAMPLES))
def test_detects_language(code):
    assert detect(SAMPLES[code])["code"] == code


def test_masking_tags_are_ignored():
    text = "[first_name_1] [last_name_1] [first_name_2] " * 10 + SAMPLES["en"]
    assert detect(text)["code"] == "en"


def test_english_with_a_spanish_stretch_is_not_plain_english():
    mixed = SAMPLES["en"] + " " + SAMPLES["es"]
    result = detect(mixed)
    assert result["code"] in ("en", "es")
    assert not is_english(result)


def test_english_with_some_chinese_is_not_plain_english():
    result = detect(SAMPLES["en"] * 2 + SAMPLES["zh"])
    assert not is_english(result)


def test_plain_english_is_english():
    assert is_english(detect(SAMPLES["en"] * 3))


def test_too_little_text_is_unknown():
    assert detect("ok yeah")["code"] == "und"


def test_short_english_excerpt_full_of_names_stays_english():
    text = ("Who didn't agree, or people who would like to. So we had Mateo, we had Sofia. "
            "Me, Diego is me, Camila, I think said yes. Valentina, I guess we don't have "
            "Valentina today. And Lucas, I'm sure we don't have Lucas here. Okay.")
    assert detect(text)["code"] == "en"
