"""Safety-net rules for Chinese, Japanese, Korean and Hindi (see pii_rules.py).

Measured on evals/masking/multilingual.py, OpenMED's multilingual model misses
a third of Chinese names and half of Korean ones, and does not mask Hindi
(Devanagari) names or cities at all. These rules add masking where the
language itself marks an identifier:

* a name after a family or role word ("我姐姐…", "제 언니 …", "मेरी बहन …");
* a name before a title or honorific ("…医生", "…先生", "… 씨", "… जी");
* a Hindi word followed by a common Indian surname ("… शर्मा");
* major cities, from short lists of each country's largest cities.

The city lists are general (the largest cities and, for Japan, the
prefectures), not the test set's; a small town is still left to the model.
Rules only add masking. Chinese and Japanese have no spaces, so the patterns
use character classes instead of word boundaries.
"""

from __future__ import annotations

import re
from typing import Callable

_HAN = r"一-鿿"
_HIRA = r"ぁ-ゖ"
_KATA = r"ァ-ヺー"
_HANGUL = r"가-힣"
_DEVA = r"ऀ-ॿ"

# --- Chinese -----------------------------------------------------------------
_ZH_SURNAMES = ("王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾肖田董袁潘"
                "于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江"
                "尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤")
_ZH_COMPOUND = "欧阳|司马|诸葛|上官|东方|皇甫|慕容|司徒|令狐|夏侯"
# Characters that end a name rather than continue it (verbs, particles).
_ZH_STOP = "住说是在和的了也都就去来给跟对把被很最还又不没让叫要会能想到从向跟这那他她它们吗呢吧啊呀"
_ZH_NAME = rf"(?:{_ZH_COMPOUND}|[{_ZH_SURNAMES}])[{_HAN}](?:(?![{_ZH_STOP}])[{_HAN}])?"
_ZH_REL = ("姐姐|妹妹|哥哥|弟弟|妈妈|爸爸|母亲|父亲|朋友|同学|同事|老公|老婆|丈夫|妻子|男朋友|女朋友|"
           "儿子|女儿|室友|老板|表姐|表妹|表哥|表弟|堂姐|堂妹|堂哥|堂弟|邻居|舅舅|阿姨|叔叔|奶奶|爷爷|"
           "外婆|外公|医生是|老师是|大夫是|叫|名字是")
_ZH_TITLE = "医生|老师|先生|女士|小姐|大夫|教授|主任|护士|律师|经理|阿姨|叔叔"
_ZH_CITIES = ("北京 上海 广州 深圳 天津 重庆 成都 武汉 杭州 南京 西安 长沙 郑州 沈阳 青岛 苏州 东莞 "
              "佛山 宁波 合肥 昆明 大连 厦门 福州 济南 哈尔滨 长春 石家庄 南昌 南宁 贵阳 太原 兰州 "
              "乌鲁木齐 呼和浩特 海口 三亚 无锡 常州 温州 珠海 中山 惠州 汕头 香港 澳门 台北 高雄 台中")

# --- Japanese ----------------------------------------------------------------
_JA_RUN = rf"[{_HAN}{_KATA}々]{{1,6}}(?:[{_HIRA}]{{2,4}}?)?"
_JA_PARTICLE = rf"(?=(?:は|が|を|に|と|も|の|から|で|へ|より)(?![{_HIRA}]))"
_JA_REL = "姉|兄|妹|弟|母|父|友達|友人|夫|妻|彼氏|彼女|息子|娘|同僚|上司|祖母|祖父|叔母|叔父|恋人"
_JA_TITLE = "さん|くん|君|ちゃん|先生|様|氏|医師|教授"
_JA_CITIES = ("東京 大阪 京都 横浜 名古屋 札幌 神戸 福岡 川崎 さいたま 広島 仙台 千葉 北九州 堺 新潟 浜松 "
              "熊本 相模原 静岡 岡山 鹿児島 那覇 金沢 長崎 奈良 姫路 宇都宮 松山 大分 "
              "北海道 青森 岩手 宮城 秋田 山形 福島 茨城 栃木 群馬 埼玉 神奈川 富山 石川 福井 山梨 "
              "長野 岐阜 愛知 三重 滋賀 兵庫 和歌山 鳥取 島根 山口 徳島 香川 愛媛 高知 佐賀 宮崎 沖縄")

# --- Korean ------------------------------------------------------------------
_KO_SURNAMES = "김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노하곽성차주우구민진나지엄채원천방공현함변염여추도소석선설마길연위표명기반왕금옥육인맹제모탁국"
_KO_PARTICLES = "이랑|에게|한테|하고|께서|은|는|이|가|을|를|와|과|랑|의|도|만|씨"
_KO_NAME = rf"[{_KO_SURNAMES}][{_HANGUL}]{{1,2}}"
_KO_REL = ("언니|오빠|누나|형|동생|엄마|아빠|어머니|아버지|친구|남편|아내|남자친구|여자친구|남친|여친|"
           "아들|딸|선배|후배|동료|사장님|팀장님|룸메이트|이웃|할머니|할아버지|이모|삼촌|고모|사촌")
_KO_TITLE = "씨|님|선생님|교수님|의사|원장님|과장님|선생"
# Role words that start with a surname syllable (선생 = teacher/doctor).
_KO_NOT = ("선생", "교수", "원장", "과장", "사장", "부장", "팀장", "의사", "간호", "부모", "하나님", "친구",
           "동생", "어머", "아버", "이야기", "정말", "진짜", "조금", "오늘", "내일", "어제", "우리")
_KO_CITIES = ("서울 부산 인천 대구 대전 광주 울산 세종 수원 고양 용인 창원 성남 청주 부천 화성 남양주 "
              "전주 천안 안산 안양 김해 평택 포항 제주 원주 춘천 강릉 경주 여수 목포 군산 구미 진주")

# --- Hindi -------------------------------------------------------------------
_HI_WORD = rf"[{_DEVA}]+"
_HI_FUNC = {"में", "से", "को", "का", "की", "के", "है", "हैं", "था", "थी", "थे", "ने", "पर", "भी", "और",
            "तो", "ही", "रहती", "रहता", "रहते", "जी", "साहब", "रोज़", "रोज", "बहुत", "अब", "कल", "आज"}
_HI_REL = ("बहन|भाई|दोस्त|सहेली|माँ|मां|मम्मी|पिता|पापा|पति|पत्नी|बेटा|बेटी|चाचा|चाची|मामा|मामी|मौसी|"
           "बुआ|दादी|दादा|नानी|नाना|पड़ोसी|बॉस|डॉक्टर|डॉ\\.?|श्री|श्रीमती|सुश्री|नाम")
_HI_SURNAMES = ("शर्मा वर्मा गुप्ता सिंह पटेल यादव जोशी मेहता कुमार अग्रवाल मिश्रा तिवारी पांडे पाण्डेय "
                "चौहान राठौर रेड्डी नायर अय्यर खान शेख़ शेख कपूर मल्होत्रा खन्ना चोपड़ा बंसल जैन शाह "
                "देसाई चौधरी ठाकुर दास बोस मुखर्जी बनर्जी चटर्जी राव पिल्लई सक्सेना श्रीवास्तव त्रिपाठी")
_HI_CITIES = ("दिल्ली मुंबई कोलकाता चेन्नई बेंगलुरु बैंगलोर हैदराबाद अहमदाबाद पुणे सूरत जयपुर लखनऊ कानपुर "
              "नागपुर इंदौर भोपाल पटना वडोदरा लुधियाना आगरा नासिक वाराणसी बनारस मेरठ राजकोट अमृतसर "
              "चंडीगढ़ गुवाहाटी रांची रायपुर देहरादून कोच्चि गोवा श्रीनगर जम्मू शिमला नोएडा गुड़गांव "
              "गुरुग्राम इलाहाबाद प्रयागराज")


def _city_rx(cities: str, before: str, after: str) -> re.Pattern:
    alts = "|".join(sorted(cities.split(), key=len, reverse=True))
    return re.compile(rf"{before}({alts}){after}")


_KO_AFTER = rf"(?=(?:{_KO_PARTICLES}|이에요|입니다)?(?![{_HANGUL}])|\s?(?:{_KO_TITLE}))"
_KO_VERBS = "만나|만났|봤|보고|좋아|사랑|알아|알고|기다|불렀|연락|전화|얘기|이야기|믿"
_ZH_VERBS = "说|告诉|觉得|问|让|知道|认为|打电话|给我"

# OpenMED often masks only half of a CJK name ("[last_name_1]洋",
# "黄[first_name_1]"); these finish the name next to its tag.
_HALF = {
    "zh": [("first_name", re.compile(rf"\[last_name_\d+\]((?![{_ZH_STOP}])[{_HAN}](?:(?![{_ZH_STOP}])[{_HAN}])?)")),
           ("last_name", re.compile(rf"((?:{_ZH_COMPOUND}|[{_ZH_SURNAMES}]))\[first_name_\d+\]"))],
    "ja": [("first_name", re.compile(rf"\[last_name_\d+\]((?:[{_HAN}々]{{1,3}}|[{_HIRA}]{{2,4}}?|[{_KATA}]{{2,6}}))"
                                     rf"(?:{_JA_PARTICLE}|(?=(?:{_JA_TITLE}|\[prefix)))"))],
    "ko": [("first_name", re.compile(rf"\[last_name_\d+\]([{_HANGUL}]{{1,2}}?){_KO_AFTER}")),
           ("last_name", re.compile(rf"(?<![{_HANGUL}])([{_HANGUL}]{{1,2}})\[first_name_\d+\]"))],
}

RULES: dict[str, list[tuple[str, re.Pattern]]] = {
    "zh": [
        ("city", _city_rx(_ZH_CITIES, "", "")),
        ("full_name", re.compile(rf"(?:^|(?<=[。，！？、,.!?\s]))({_ZH_NAME})(?:{_ZH_VERBS})")),
        ("full_name", re.compile(rf"(?:我和|我跟|和我|跟我)({_ZH_NAME})")),
        ("full_name", re.compile(rf"(?:{_ZH_REL})[，, ]?({_ZH_NAME})")),
        ("full_name", re.compile(rf"({_ZH_NAME})(?:{_ZH_TITLE})")),
    ],
    "ja": [
        ("city", _city_rx(_JA_CITIES, "", "")),
        ("full_name", re.compile(rf"(?:{_JA_REL})の({_JA_RUN}){_JA_PARTICLE}")),
        ("full_name", re.compile(rf"(?<![{_HAN}{_KATA}々])((?:[{_HAN}々]{{1,4}}|[{_KATA}]{{2,8}}))(?:{_JA_TITLE})")),
    ],
    "ko": [
        ("city", _city_rx(_KO_CITIES, rf"(?<![{_HANGUL}])", rf"(?=(?:에서|에|은|는|이|가|을|를|로|으로|의|까지|부터|도|만)?(?![{_HANGUL}]))")),
        ("full_name", re.compile(rf"(?:{_KO_REL})\s+({_KO_NAME}?)(?=(?:{_KO_PARTICLES})?(?![{_HANGUL}]))")),
        ("full_name", re.compile(rf"(?<![{_HANGUL}])({_KO_NAME})\s?(?=(?:{_KO_TITLE}))")),
        ("full_name", re.compile(rf"(?<![{_HANGUL}])({_KO_NAME}?)(?:을|를|이랑|랑|하고|와|과|한테|에게)\s+(?:{_KO_VERBS})")),
    ],
    "hi": [
        ("city", _city_rx(_HI_CITIES, rf"(?<![{_DEVA}])", rf"(?![{_DEVA}])")),
        ("full_name", re.compile(rf"(?<![{_DEVA}])({_HI_WORD}\s+(?:{'|'.join(_HI_SURNAMES.split())}))(?![{_DEVA}])")),
        ("full_name", re.compile(rf"(?<![{_DEVA}])(?:{_HI_REL})\s+({_HI_WORD}(?:\s+{_HI_WORD})?)(?![{_DEVA}])")),
        ("full_name", re.compile(rf"(?<![{_DEVA}])({_HI_WORD})\s+(?:जी|साहब)(?![{_DEVA}])")),
    ],
}
_TAG = re.compile(r"\[[a-z_]+(?:_\d+)?\]")


def _trim_hindi(value: str) -> str:
    """Drop function words a Hindi name pattern swallowed ("प्रिया में" -> "प्रिया")."""
    words = value.split()
    while words and words[-1] in _HI_FUNC:
        words.pop()
    return " ".join(words)


def apply(masked: str, lang: str, number: Callable[[str, str], int], key: Callable[[str], str],
          found: list | None = None) -> tuple[str, dict[str, int]]:
    """Mask what the ``lang`` rules find; same contract as pii_rules.apply."""
    counts: dict[str, int] = {}
    for label, rx in _HALF.get(lang, []) + RULES.get(lang, []):
        def _sub(m: re.Match) -> str:
            value = m.group(1)
            if not value or _TAG.search(value):
                return m.group(0)
            if lang == "ko" and value.startswith(_KO_NOT):
                return m.group(0)
            if lang == "hi":
                value = _trim_hindi(value)
                if not value or value.split()[0] in _HI_FUNC:
                    return m.group(0)
            n = number(label, key(value))
            counts[label] = counts.get(label, 0) + 1
            if found is not None:
                found.append((value, f"[{label}_{n}]"))
            start = m.start(1) - m.start(0)
            whole = m.group(0)
            return whole[:start] + f"[{label}_{n}]" + whole[start + len(value):]
        masked = rx.sub(_sub, masked)
    return masked, counts


# Scripts without capital letters: an identifier is propagated when it has at
# least two characters of one of these scripts.
SCRIPT = re.compile(rf"[{_HAN}{_HIRA}{_KATA}{_HANGUL}{_DEVA}]")
