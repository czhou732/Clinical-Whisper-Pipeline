#!/usr/bin/env python3
"""Masking recall in Chinese, Japanese, Korean, Hindi and Spanish.

Interview-style sentences with invented names and real major cities, in each
language's own script. Scored by character, since Chinese and Japanese do not
separate words with spaces: an identifier counts as caught when every one of
its characters ends up inside a tag.

    python evals/masking/multilingual.py [--no-safety-net] [--held-out] [--second-check]
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SETS = {
    "zh": {
        "names": ["王丽", "张伟", "李娜", "刘洋", "陈静", "杨帆", "赵敏", "黄磊", "周婷", "吴昊"],
        "cities": ["深圳", "上海", "北京", "成都", "武汉", "杭州", "广州", "西安", "南京", "长沙"],
        "templates": ["我姐姐{N}住在{C}，她每天都给我打电话。", "我的医生是{N}。", "我和{N}是在{C}认识的。",
                      "我从{C}搬过来已经三年了。", "我最好的朋友{N}最近不太理我。", "{N}说她下个月去{C}。"],
    },
    "ja": {
        "names": ["田中美咲", "佐藤健", "鈴木花子", "高橋翔", "伊藤さくら", "渡辺大輔", "山本結衣", "中村蓮"],
        "cities": ["大阪", "東京", "京都", "札幌", "福岡", "名古屋", "神戸", "横浜"],
        "templates": ["姉の{N}は{C}に住んでいます。", "主治医は{N}先生です。", "{C}で{N}と知り合いました。",
                      "去年{C}から引っ越しました。", "友達の{N}が毎日電話をくれます。"],
    },
    "ko": {
        "names": ["김민지", "이서준", "박지은", "최현우", "정수빈", "강도윤", "조하은", "윤재민"],
        "cities": ["부산", "서울", "인천", "대구", "대전", "광주", "울산", "수원"],
        # ~는/~를/~가: the particle agrees with the name's last syllable (은/는, 을/를, 이/가).
        "templates": ["제 언니 {N}~는 {C}에 살아요.", "담당 의사는 {N} 선생님이에요.", "{C}에서 {N}~를 만났어요.",
                      "작년에 {C}에서 이사 왔어요.", "친구 {N}~가 매일 전화해요."],
    },
    "hi": {
        "names": ["प्रिया शर्मा", "राहुल वर्मा", "अनीता सिंह", "विक्रम पटेल", "सुनीता यादव", "अर्जुन मेहता",
                  "कविता गुप्ता", "रोहित जोशी"],
        "cities": ["पुणे", "मुंबई", "दिल्ली", "जयपुर", "लखनऊ", "पटना", "कोलकाता", "हैदराबाद"],
        "templates": ["मेरी बहन {N} {C} में रहती है।", "मेरे डॉक्टर {N} हैं।", "मैं {N} से {C} में मिला था।",
                      "मैं पिछले साल {C} से आया हूँ।", "मेरा दोस्त {N} रोज़ फोन करता है।"],
    },
    "es": {
        "names": ["María López", "Javier Ramírez", "Lucía Fernández", "Diego Morales", "Camila Torres",
                  "Mateo Herrera", "Valentina Ruiz", "Andrés Castillo"],
        "cities": ["Guadalajara", "Monterrey", "Sevilla", "Bogotá", "Lima", "Valencia", "Medellín", "Quito"],
        "templates": ["Mi hermana {N} vive en {C}.", "Mi médico es el doctor {N}.", "Conocí a {N} en {C}.",
                      "Me mudé de {C} el año pasado.", "Mi amiga {N} me llama todos los días."],
    },
}

# Written after the language rules, never used to tune them: the honest test.
# New names and sentence frames; the last three cities of each list are
# smaller places deliberately left out of pii_rules_intl.py's city lists.
HELD_OUT = {
    "zh": {
        "names": ["孙悦", "马超", "朱琳", "胡斌", "郭敏", "何晴", "高翔", "林峰", "罗薇", "郑凯"],
        "cities": ["苏州", "厦门", "昆明", "常德", "岳阳", "义乌"],
        "templates": ["上个星期{N}来看我了。", "{N}是我以前的同事。", "我们是在{C}长大的。", "我妈妈叫{N}。",
                      "我跟{N}吵了一架。", "护士{N}对我很好。", "那时候我住在{C}，后来搬走了。"],
    },
    "ja": {
        "names": ["小林陽菜", "加藤悠真", "吉田葵", "山田拓海", "佐々木凛", "井上陽向", "木村杏", "斎藤蒼"],
        "cities": ["仙台", "広島", "金沢", "函館", "倉敷", "豊橋"],
        "templates": ["{N}さんとは高校からの付き合いです。", "{C}出身です。", "昨日{N}に会いました。",
                      "母は{N}といいます。", "{C}の病院に入院していました。", "同僚の{N}が心配してくれました。"],
    },
    "ko": {
        "names": ["한지민", "오지훈", "서예린", "신동현", "권나연", "황민재", "안소희", "송태양"],
        "cities": ["수원", "전주", "포항", "김천", "통영", "속초"],
        "templates": ["{N}~가 어제 집에 왔어요.", "저는 {C} 출신이에요.", "우리 엄마는 {N} 씨예요.",
                      "{N}하고 많이 싸웠어요.", "간호사 {N} 씨가 친절했어요.", "{C}에 있는 병원에 입원했었어요.",
                      "동료 {N}~가 걱정해 줬어요."],
    },
    "hi": {
        "names": ["नेहा कपूर", "अमित कुमार", "पूजा मिश्रा", "संजय तिवारी", "रीना चौधरी", "मनोज राव", "सीमा", "रवि"],
        "cities": ["भोपाल", "इंदौर", "चंडीगढ़", "अजमेर", "उज्जैन", "हल्द्वानी"],
        "templates": ["कल {N} मुझसे मिलने आई थी।", "मैं {C} में पैदा हुआ था।", "मेरी माँ का नाम {N} है।",
                      "{N} जी ने मुझे दवाई दी।", "मैंने {N} से बहुत झगड़ा किया।", "मेरे पड़ोसी {N} बहुत मदद करते हैं।",
                      "हम {C} के एक अस्पताल में थे।"],
    },
    "es": {
        "names": ["Sofía Navarro", "Pablo Ortega", "Elena Vargas", "Tomás Rojas", "Inés", "Raúl"],
        "cities": ["Zaragoza", "Cuenca", "Arequipa", "Rosario", "Tarija", "Jalapa"],
        "templates": ["Ayer vino {N} a verme.", "Soy de {C}.", "Mi mamá se llama {N}.",
                      "La enfermera {N} fue muy amable.", "Estuve internado en un hospital de {C}."],
    },
}

# Written Oct 2 2026, after every rule and bug fix above and before any run
# on it: the replication set. New names, new frames (several with no cue word
# around the name), and the last three cities of each list again outside the
# rules' city lists.
HELD_OUT_2 = {
    "zh": {
        "names": ["宋佳", "唐亮", "许静怡", "韩雪", "冯磊", "曹宇", "彭丽", "谢天明", "邓芳", "蒋涛"],
        "cities": ["天津", "青岛", "大连", "绵阳", "柳州", "赣州"],
        "templates": ["昨天晚上{N}给我发了很多消息。", "我们家以前在{C}开了个小店。", "后来{N}也不太联系了。",
                      "他叫{N}，是我大学室友。", "那次去{C}看病花了很多钱。", "{N}和我都觉得很累。",
                      "我老公{N}一直劝我来这里。"],
    },
    "ja": {
        "names": ["森本大和", "石川美優", "前田颯", "藤田結菜", "岡田蒼空", "長谷川芽依", "村上樹", "近藤さくら"],
        "cities": ["横浜", "神戸", "新潟", "釧路", "四日市", "八戸"],
        "templates": ["昨日の夜、{N}から電話がありました。", "実家は{C}にあります。", "{N}にはまだ話していません。",
                      "夫の{N}は仕事が忙しいです。", "{C}の大学に通っていました。", "担当の{N}さんに相談しました。"],
    },
    "ko": {
        "names": ["임하늘", "한서윤", "오민호", "배수아", "백지훈", "허윤서", "남궁민", "유도현"],
        "cities": ["인천", "청주", "제주", "여주", "거제", "삼척"],
        "templates": ["어젯밤에 {N}~가 문자를 많이 보냈어요.", "본가는 {C}에 있어요.", "{N}한테는 아직 말 안 했어요.",
                      "남편 {N}~는 요즘 바빠요.", "{C}에 있는 대학교를 다녔어요.", "상담 선생님 {N} 씨한테 얘기했어요."],
    },
    "hi": {
        "names": ["आरती सक्सेना", "विनोद त्रिपाठी", "स्नेहा बनर्जी", "राजेश नायर", "मीना", "सुरेश", "अनुराधा देसाई", "करण"],
        "cities": ["लखनऊ", "नागपुर", "कोच्चि", "बरेली", "झांसी", "सतना"],
        "templates": ["कल रात {N} ने मुझे बहुत मैसेज किए।", "हमारा घर {C} में है।", "{N} को मैंने अभी तक नहीं बताया।",
                      "मेरे पति {N} आजकल बहुत व्यस्त हैं।", "मैं {C} के कॉलेज में पढ़ता था।", "मैंने काउंसलर {N} से बात की।"],
    },
    "es": {
        "names": ["Gabriela Méndez", "Joaquín Salazar", "Paula Guerrero", "Emilio Cabrera", "Rocío", "Nicolás"],
        "cities": ["Barcelona", "Puebla", "Cali", "Ensenada", "Cajamarca", "Rancagua"],
        "templates": ["Anoche {N} me mandó muchos mensajes.", "Mi familia vive en {C}.", "A {N} todavía no le he dicho nada.",
                      "Mi esposo {N} anda muy ocupado.", "Estudié en una universidad de {C}.", "Hablé con la consejera {N}."],
    },
}

_PARTICLES = {"는": "은", "를": "을", "가": "이"}


def _josa(text: str) -> str:
    """Resolve "~는" to 은 or 는 by whether the syllable before it ends in a consonant."""
    def pick(m: re.Match) -> str:
        final = (ord(m.group(1)) - 0xAC00) % 28 != 0
        return m.group(1) + (_PARTICLES[m.group(2)] if final else m.group(2))
    return re.sub(r"([\uac00-\ud7a3])~(는|를|가)", pick, text)


def build(lang: str, rounds: int = 12, seed: int = 7, sets: dict | None = None) -> list[dict]:
    rng = random.Random(f"{seed}-{lang}")
    spec, rows = (sets or SETS)[lang], []
    for k in range(rounds):
        for t_idx, t in enumerate(spec["templates"]):
            text, ents, i = "", [], 0
            values = {"N": (rng.choice(spec["names"]), "name"), "C": (rng.choice(spec["cities"]), "city")}
            while i < len(t):
                if t[i] == "{":
                    v, typ = values[t[i + 1]]
                    ents.append({"start": len(text), "end": len(text) + len(v), "type": typ})
                    text += v
                    i += 3
                elif t[i] == "~":
                    text = _josa(text + t[i:i + 2])
                    i += 2
                else:
                    text += t[i]
                    i += 1
            rows.append({"id": f"{lang}{t_idx}_{k}", "text": text, "entities": ents})
    return rows


def caught_chars(raw: str, masked: str) -> set[int]:
    """Indices of ``raw`` characters that were replaced by tags."""
    plain = re.sub(r"\[[A-Za-z_]+(?:_\d+)?\]", "\0", masked)
    out: set[int] = set()
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=raw, b=plain, autojunk=False).get_opcodes():
        if op == "delete" or (op == "replace" and "\0" in plain[j1:j2]):
            out.update(range(i1, i2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-safety-net", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--held-out", action="store_true", help="the set written after the rules")
    ap.add_argument("--held-out-2", action="store_true", help="the replication set written after all fixes")
    ap.add_argument("--threshold", type=float, default=None, help="default: the app's")
    ap.add_argument("--model", default=None, help="compare another OpenMED model (default: the app's choice)")
    ap.add_argument("--langs", default=",".join(SETS))
    ap.add_argument("--second-check", action="store_true",
                    help="add the scoring model's name check (name_sweep.py), one sentence at a time")
    args = ap.parse_args()
    sets = HELD_OUT_2 if args.held_out_2 else HELD_OUT if args.held_out else SETS
    import inference_pipeline as ip
    import name_sweep
    from pii_scrubber import PIIScrubber

    ask = name_sweep.llama(__import__("addons").SCORING_MODEL) if args.second_check else None

    report = {}
    for lang in args.langs.split(","):
        kwargs = ip._masker_for({"code": lang, "name": lang, "confidence": 1.0, "other_share": 0.0})
        if args.model:
            kwargs = {**kwargs, "model_name": args.model}
            kwargs.pop("cache_dir", None)
        if args.threshold is not None:
            kwargs["confidence_threshold"] = args.threshold
        scrubber = PIIScrubber(**{**kwargs, "safety_net": not args.no_safety_net})
        hit, over, total_other = defaultdict(lambda: [0, 0]), 0, 0
        for r in build(lang, sets=sets):
            scrubber._ids = {}
            masked = scrubber.scrub_text(r["text"])
            if ask is not None:
                # One sentence per request: no help from the same name elsewhere.
                masked = name_sweep.mask([masked], [r["text"]], ask, scrubber._number)[0][0]
            got = caught_chars(r["text"], masked)
            ent_chars = set()
            for e in r["entities"]:
                ent_chars.update(range(e["start"], e["end"]))
                need = [c for c in range(e["start"], e["end"]) if not r["text"][c].isspace()]
                hit[e["type"]][0] += all(c in got for c in need)
                hit[e["type"]][1] += 1
            others = [c for c in range(len(r["text"])) if c not in ent_chars and not r["text"][c].isspace()]
            total_other += len(others)
            over += sum(c in got for c in others)
        n = sum(v[1] for v in hit.values())
        report[lang] = {"model": kwargs.get("model_name", "English default"),
                        "threshold": scrubber.confidence_threshold, "second_check": bool(ask),
                        "recall": round(sum(v[0] for v in hit.values()) / n, 3),
                        "by_type": {k: round(v[0] / v[1], 3) for k, v in hit.items()},
                        "over_masking_chars": round(over / max(total_other, 1), 3)}
        print(lang, json.dumps(report[lang]), flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
