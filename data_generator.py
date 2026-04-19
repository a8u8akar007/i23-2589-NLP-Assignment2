import os
import random

# 1. Defined Gazetteer (Urdu)
NAMES = ["عمران", "ابوبکر", "علی", "احمد", "محمود", "فاطمہ", "سارہ", "عائشہ", "حمزہ", "عثمان", "بلال", "حسن", "زاہد", "راشد", "ساجد", "عامر", "ناصر", "یاسر", "خالد", "طارق", "عمر", "اسد", "فیصل", "جاوید", "اقبال", "نواز", "شہباز", "مریم", "کلثوم", "بشری", "فواد", "سمیع", "جنید", "عاطف", "راحت", "عابد", "فرح", "صائمہ", "نزہت", "ریحانہ", "کشور", "شائستہ", "انجم", "شبنم", "بلقیس", "طلعت", "طارق", "ارشد", "سجاد", "عرفان"]
LOCATIONS = ["پاکستان", "لاہور", "کراچی", "اسلام آباد", "پشاور", "کوئٹہ", "فیصل آباد", "ملتان", "گوجرانوالہ", "سیالکوٹ", "حیدرآباد", "سکھر", "لاڑکانہ", "مری", "گلگت", "بلتستان", "بھارت", "چین", "امریکہ", "برطانیہ", "لندن", "پیرس", "دبئی", "کابل", "تہران", "دہلی", "ٹوکیو", "مکہ", "مدینہ", "سعودی عرب", "ایران", "ترکی", "مصر", "جرمنی", "فرانس", "اٹلی", "کینیڈا", "آسٹریلیا", "جاپان", "ماسکو", "بیجنگ", "نیویارک", "واشنگٹن", "برلن", "سری لنکا", "بنگلہ دیش", "روس", "پنجاب", "سندھ", "خیبر"]
ORGS = ["پی ٹی آئی", "نون لیگ", "پی پی پی", "فوج", "عدالت", "حکومت", "اقوام متحدہ", "فیفا", "پی سی بی", "اسٹیٹ بینک", "نادرا", "واسا", "پولیس", "نیب", "ایف آئی اے", "سپریم کورٹ", "ہائیکورٹ", "اسمبلی", "سینیٹ", "کالج", "یونیورسٹی", "ہسپتال", "تنظیم", "تحریک", "وفاق", "جماعت", "ایجنسی", "کمسیون", "یونین", "ادارہ"]

# 2. POS Rules
VERBS = ["ہے", "تھا", "ہیں", "تھے", "کر", "گیا", "ہوا", "رہی", "رہا", "تھی", "آئی", "سکا", "کرنے", "کہا", "ہوئے", "ہو"]
POSTPOSITIONS = ["کے", "میں", "کی", "سے", "کا", "نے", "کو", "پر", "لیے", "ساتھ"]
ADJECTIVES = ["بڑا", "چھوٹا", "اچھا", "برا", "نیا", "پرانا", "زیادہ", "کم", "پہلی", "دوسری", "اعلی", "بہتر"]
PRONOUNS = ["اس", "ان", "وہ", "جس", "انھوں", "یہ", "اپنے", "آئی", "تم", "آپ", "میری"]
CONJUNCTIONS = ["اور", "کہ", "لیکن", "یا", "مگر", "بلکہ", "چنانچہ"]
PARTICLES = ["بھی", "ہی", "تو", "نہ", "نا"]

def get_pos_tag(word):
    if word in NAMES or word in LOCATIONS or word in ORGS: return "PN"
    if word in VERBS: return "V"
    if word in POSTPOSITIONS: return "P"
    if word in ADJECTIVES: return "ADJ"
    if word in PRONOUNS: return "PR"
    if word in CONJUNCTIONS: return "CONJ"
    if word in PARTICLES: return "PART"
    if any(char.isdigit() for char in word) or word in ["ایک", "دو", "تین", "چار", "پانچ"]: return "NUM"
    if word in ["۔", "،", "!", "?", "؛"]: return "PUNCT"
    return "N"

def get_ner_tag(word, is_start):
    tag = "O"
    cat = ""
    if word in NAMES: cat = "PER"
    elif word in LOCATIONS: cat = "LOC"
    elif word in ORGS: cat = "ORG"
    
    if cat:
        return f"B-{cat}" if is_start else f"I-{cat}"
    return "O"

# 3. Process Sentences
with open("data/cleaned.txt", "r", encoding="utf-8") as f:
    text = f.read()
tokens = text.split()

# Create 500 fake sentences by grouping tokens (since we don't have sentence markers)
sentences = []
for i in range(0, min(len(tokens), 7500), 15): # 15 words per sentence
    sentences.append(tokens[i:i+15])

sentences = sentences[:500]
random.shuffle(sentences)
split = 400

# 4. Generate Files
def write_conll(filename, data, mode="POS"):
    with open(filename, "w", encoding="utf-8") as f:
        for sent in data:
            for i, word in enumerate(sent):
                if mode == "POS":
                    tag = get_pos_tag(word)
                else:
                    tag = get_ner_tag(word, True) # Simplifying for rule-based BIO
                f.write(f"{word}\t{tag}\n")
            f.write("\n")

write_conll("data/pos_train.conll", sentences[:split], "POS")
write_conll("data/pos_test.conll", sentences[split:], "POS")
write_conll("data/ner_train.conll", sentences[:split], "NER")
write_conll("data/ner_test.conll", sentences[split:], "NER")

print("Created 500 sentences across train/test CoNLL files in data/ folder.")
