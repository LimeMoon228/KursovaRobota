from ufal.udpipe import Model, Pipeline


model_path = "ukrainian-iu-ud-2.5-191206.udpipe"
model = Model.load(model_path)

pipeline = Pipeline(model, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")

POS_MAP = {
    'NOUN': 1, 'VERB': 2, 'ADV': 3, 'ADJ': 4, 'PRON': 5, 'ADP': 6,
    'CCONJ': 7, 'DET': 8, 'NUM': 9, 'PART': 10, 'PUNCT': 11, 'X': 12
}

CASE_MAP = {
    'Nom': 1, 'Gen': 2, 'Dat': 3, 'Acc': 4, 'Ins': 5, 'Loc': 6, 'Voc': 7, 'N/A': 0
}

NUMBER_MAP = {
    'Sing': 1, 'Plur': 2, 'N/A': 0
}

GENDER_MAP = {
    'Masc': 1, 'Fem': 2, 'Neut': 3, 'N/A': 0
}

DEP_MAP = {
    'nsubj': 1,
    'root': 2,
    'obj': 3,
    'obl': 4,
    'nmod': 5,
    'amod': 6,
    'advmod:place': 7,
    'advmod:time': 8,
    'advmod:manner': 9,
    'punct': 10,
    'case': 11,
    'cc': 12,
    'conj': 13,
    'det': 14,
    'unknown': 0
}


def analyze_text(text):
    processed = pipeline.process(text)
    word_vectors = []

    lines = processed.split('\n')
    tokens = [line.split('\t') for line in lines if line.strip() and not line.startswith('#')]

    saw_conjunction = False
    prev_dep_code = None

    for idx, fields in enumerate(tokens):
        if len(fields) < 10:
            continue

        if '-' in fields[0] or '.' in fields[0]:
            continue

        word_idx = int(fields[0])
        word = fields[1]
        lemma = fields[2]
        upos = fields[3]
        feats = fields[5]
        head = int(fields[6])
        deprel = fields[7]

        case, number, gender = 'N/A', 'N/A', 'N/A'
        if feats != '_':
            feat_pairs = feats.split('|')
            for feat in feat_pairs:
                if 'Case=' in feat:
                    case = feat.split('=')[1]
                if 'Number=' in feat:
                    number = feat.split('=')[1]
                if 'Gender=' in feat:
                    gender = feat.split('=')[1]

        pos_code = POS_MAP.get(upos, 12)
        case_code = CASE_MAP.get(case, 0)
        number_code = NUMBER_MAP.get(number, 0)
        gender_code = GENDER_MAP.get(gender, 0)


        if upos == 'CCONJ' and deprel == 'cc':
            saw_conjunction = True

        if saw_conjunction and deprel == 'conj':
            if upos == 'VERB':
                deprel = 'root'
            elif upos == 'NOUN':
                if idx < len(tokens) - 1 and tokens[idx + 1][3] == 'VERB':
                    deprel = 'nsubj'
                else:
                    deprel = 'obj'

        dep_code = DEP_MAP.get(deprel, 0)

        has_prev_unknown = 1 if (prev_dep_code == 11 and word_idx > 1) else 0

        vector = [
            word_idx, pos_code, case_code, number_code, gender_code,
            head, abs(head - word_idx), len(lemma),
            has_prev_unknown, dep_code
        ]
        word_vectors.append(vector)

        prev_dep_code = dep_code

    return word_vectors


def save_to_file(vectors, filename="ukrainian_analysis.txt"):
    with open(filename, 'w', encoding='utf-8') as f:
        for v in vectors:
            line = f"{v}\n"
            f.write(line)


with open("text.txt", 'r', encoding='utf-8') as text:
    ukrainian_text = text.read()

vectors = analyze_text(ukrainian_text)
save_to_file(vectors)
print("Анализ сохранён в файл 'ukrainian_analysis.txt'")