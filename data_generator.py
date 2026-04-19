import os
import random

# 1. Defined Gazetteer (Urdu)
NAMES = ["عمران خان", "شہباز شریف", "مریم نواز", "عمران", "ابوبکر", "علی", "احمد", "محمود", "فاطمہ", "سارہ", "عائشہ", "حمزہ", "عثمان", "بلال", "حسن", "زاہد", "راشد", "ساجد", "عامر", "ناصر", "یاسر", "خالد", "طارق", "عمر", "اسد", "فیصل", "جاوید", "اقبال", "نواز", "مریم", "کلثوم", "بشری", "فواد", "سمیع", "جنید", "عاطف", "راحت", "عابد", "فرح", "صائمہ", "نزہت", "ریحانہ", "کشور", "شائستہ", "انجم", "شبنم", "بلقیس", "طلعت", "ارشد", "سجاد", "عرفان"]
LOCATIONS = ["فیصل آباد", "اسلام آباد", "سعودی عرب", "پاکستان", "لاہور", "کراچی", "پشاور", "کوئٹہ", "فیصل آباد", "ملتان", "گوجرانوالہ", "سیالکوٹ", "حیدرآباد", "سکھر", "لاڑکانہ", "مری", "گلگت", "بلتستان", "بھارت", "چین", "امریکہ", "برطانیہ", "لندن", "پیرس", "دبئی", "کابل", "تہران", "دہلی", "ٹوکیو", "مکہ", "مدینہ", "ایران", "ترکی", "مصر", "جرمنی", "فرانس", "اٹلی", "کینیڈا", "آسٹریلیا", "جاپان", "ماسکو", "بیجنگ", "نیویارک", "واشنگٹن", "برلن", "سری لنکا", "بنگلہ دیش", "روس", "پنجاب", "سندھ", "خیبر"]
ORGS = ["لاہور پولیس", "اقوام متحدہ", "پی ٹی آئی", "نون لیگ", "پی پی پی", "فوج", "عدالت", "حکومت", "فیفا", "پی سی بی", "اسٹیٹ بینک", "نادرا", "واسا", "پولیس", "نیب", "ایف آئی اے", "سپریم کورٹ", "ہائیکورٹ", "اسمبلی", "سینیٹ", "کالج", "یونیورسٹی", "ہسپتال", "تنظیم", "تحریک", "وفاق", "جماعت", "ایجنسی", "کمسیون", "یونین", "ادارہ"]

# 2. POS Rules
VERBS = ["ہے", "تھا", "ہیں", "تھے", "کر", "گیا", "ہوا", "رہی", "رہا", "تھی", "آئی", "سکا", "کرنے", "کہا", "ہوئے", "ہو"]
POSTPOSITIONS = ["کے", "میں", "کی", "سے", "کا", "نے", "کو", "پر", "لیے", "ساتھ"]
ADJECTIVES = ["بڑا", "چھوٹا", "اچھا", "برا", "نیا", "پرانا", "زیادہ", "کم", "پہلی", "دوسری", "اعلی", "بہتر"]
PRONOUNS = ["اس", "ان", "وہ", "جس", "انھوں", "یہ", "اپنے", "آئی", "تم", "آپ", "میری"]
CONJUNCTIONS = ["اور", "کہ", "لیکن", "یا", "مگر", "بلکہ", "چنانچہ"]
PARTICLES = ["بھی", "ہی", "تو", "نہ", "نا"]
ADVERBS = ["بہت", "تقریباً", "فوراً", "آہستہ", "تیز", "شاید", "ابھی", "ہمیشہ", "روزانہ"]
DETERMINERS = ["یہ", "وہ", "اسی", "اسی", "کئی", "کچھ", "ہر", "کونسا", "کونسی"]
QUES = ["کیا", "کب", "کون", "کیسے", "کیوں", "کتنا"]

def get_pos_tag(word):
    if word in NAMES or word in LOCATIONS or word in ORGS: return "PN"
    if word in VERBS: return "V"
    if word in POSTPOSITIONS: return "P"
    if word in ADJECTIVES: return "ADJ"
    if word in PRONOUNS: return "PR"
    if word in ADVERBS: return "ADV"
    if word in DETERMINERS: return "DET"
    if word in QUES: return "QUES"
    if word in CONJUNCTIONS: return "CONJ"
    if word in PARTICLES: return "PART"
    if any(char.isdigit() for char in word) or word in ["ایک", "دو", "تین", "چار", "پانچ"]: return "NUM"
    if word in ["۔", "،", "!", "?", "؛"]: return "PUNCT"
    return "N"

# 3. Process Sentences
with open("data/cleaned.txt", "r", encoding="utf-8") as f:
    text = f.read()
tokens = text.split()

sentences = []
for i in range(0, min(len(tokens), 7500), 15):
    sentences.append(tokens[i:i+15])

sentences = sentences[:500]
random.shuffle(sentences)

# 70/15/15 Split
train_idx = int(0.70 * len(sentences))
val_idx = int(0.85 * len(sentences))

train_sents = sentences[:train_idx]
val_sents = sentences[train_idx:val_idx]
test_sents = sentences[val_idx:]

# 4. Generate Files with BIO tagging
def write_conll(filename, data, mode="POS"):
    with open(filename, "w", encoding="utf-8") as f:
        for sent in data:
            if mode == "POS":
                for word in sent:
                    f.write(f"{word}\t{get_pos_tag(word)}\n")
            else:
                sent_text = " ".join(sent)
                tags = ["O"] * len(sent)
                
                # Check for each entity type
                for cat, gazetteer in [("PER", NAMES), ("LOC", LOCATIONS), ("ORG", ORGS)]:
                    for entry in gazetteer:
                        entity_tokens = entry.split()
                        n = len(entity_tokens)
                        for i in range(len(sent) - n + 1):
                            if sent[i:i+n] == entity_tokens:
                                tags[i] = f"B-{cat}"
                                for j in range(1, n):
                                    tags[i+j] = f"I-{cat}"
                
                for word, tag in zip(sent, tags):
                    f.write(f"{word}\t{tag}\n")
            f.write("\n")

write_conll("data/pos_train.conll", train_sents, "POS")
write_conll("data/pos_val.conll", val_sents, "POS")
write_conll("data/pos_test.conll", test_sents, "POS")

write_conll("data/ner_train.conll", train_sents, "NER")
write_conll("data/ner_val.conll", val_sents, "NER")
write_conll("data/ner_test.conll", test_sents, "NER")

print(f"Created 500 sentences: {len(train_sents)} Train, {len(val_sents)} Val, {len(test_sents)} Test.")
