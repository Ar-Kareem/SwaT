from functools import lru_cache
import re
import string
import numpy as np

SUPPORTED_METRICS = frozenset(('vqa', 'gqa_exact', 'chartqa_relaxed', 'anls', 'mcq',
    'aokvqa_da', 'normalized_exact', 'binary_prefix', 'pope_accuracy',
    'mmmu_open_accuracy', 'mme_accuracy', 'mathvision_accuracy'))


def validate_metrics(rows):
    rows = list(rows)
    metrics = {row['metric'] for row in rows}
    unknown = metrics - SUPPORTED_METRICS
    if unknown:
        raise ValueError(f'Unsupported scoring metrics: {", ".join(sorted(unknown))}')
    if 'mathvision_accuracy' in metrics:
        try:
            from latex2sympy2_extended import latex2sympy
        except ImportError as exc:
            raise ValueError('MATH-Vision requires latex2sympy2_extended==1.11.0; install requirements.txt') from exc
    for row in rows:
        try:
            if not row['answers']:
                raise ValueError('No reference answers')
            if row['metric'] == 'mme_accuracy':
                if not row['pair_id'] or not row['question_type']:
                    raise ValueError('Missing MME pair/category metadata')
            if row['metric'] == 'mcq':
                choices = row.get('choices', dict.fromkeys('ABCD'))
                if not isinstance(choices, dict) or row['correct_choice'] not in choices:
                    raise ValueError('Invalid multiple-choice labels')
            for output in ('', str(row['answers'][0]['answer'])):
                value = score_answer(output, row)
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError('Score must be finite and between zero and one')
        except Exception as exc:
            raise ValueError(f'Invalid scoring input for question {row.get("question_id", "?")} '
                             f'({row["metric"]}): {exc}') from exc


def mme_prediction(output):
    output = output.lower()
    if output in ('yes', 'no'):
        return output
    return 'yes' if 'yes' in output[:4] else 'no' if 'no' in output[:4] else 'other'


def mme_summary(records, rows):
    from collections import defaultdict
    lookup = {str(row['question_id']): row for row in rows}
    categories = defaultdict(lambda: defaultdict(list))
    for record in records:
        row = lookup[str(record['question_id'])]
        categories[row['question_type']][row['pair_id']].append(record['score'])
    result = {}
    for category, pairs in sorted(categories.items()):
        values = [v for pair in pairs.values() for v in pair]
        complete_pairs = [pair for pair in pairs.values() if len(pair) == 2]
        acc = sum(values) / len(values)
        acc_plus = sum(all(pair) for pair in complete_pairs) / len(complete_pairs) if complete_pairs else None
        result[category] = {'accuracy': 100 * acc,
            'accuracy_plus': 100 * acc_plus if acc_plus is not None else None,
            'score': 100 * (acc + acc_plus) if acc_plus is not None else None,
            'questions': len(values), 'complete_pairs': len(complete_pairs),
            'incomplete_pairs': len(pairs) - len(complete_pairs)}
    sections = {
        'perception': ('existence', 'count', 'position', 'color', 'posters', 'celebrity', 'scene', 'landmark', 'artwork', 'OCR'),
        'cognition': ('commonsense_reasoning', 'numerical_calculation', 'text_translation', 'code_reasoning')}
    totals = {name: sum(result[c]['score'] for c in cats)
              if all(c in result and result[c]['score'] is not None and not result[c]['incomplete_pairs'] for c in cats)
              else None for name, cats in sections.items()}
    return {'categories': result, **totals}

class _VQANormalizer:

    def __init__(self):
        self.contractions = {'aint': "ain't", 'arent': "aren't", 'cant': "can't", 'couldve': "could've", 'couldnt': "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've", 'didnt': "didn't", 'doesnt': "doesn't", 'dont': "don't", 'hadnt': "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", 'hasnt': "hasn't", 'havent': "haven't", 'hed': "he'd", "hed've": "he'd've", "he'dve": "he'd've", 'hes': "he's", 'howd': "how'd", 'howll': "how'll", 'hows': "how's", "Id've": "I'd've", "I'dve": "I'd've", 'Im': "I'm", 'Ive': "I've", 'isnt': "isn't", 'itd': "it'd", "itd've": "it'd've", "it'dve": "it'd've", 'itll': "it'll", "let's": "let's", 'maam': "ma'am", 'mightnt': "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", 'mightve': "might've", 'mustnt': "mustn't", 'mustve': "must've", 'neednt': "needn't", 'notve': "not've", 'oclock': "o'clock", 'oughtnt': "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", 'shant': "shan't", "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's", 'shouldve': "should've", 'shouldnt': "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", "somebody'd": 'somebodyd', "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", 'somebodyll': "somebody'll", 'somebodys': "somebody's", 'someoned': "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", 'someonell': "someone'll", 'someones': "someone's", 'somethingd': "something'd", "somethingd've": "something'd've", "something'dve": "something'd've", 'somethingll': "something'll", 'thats': "that's", 'thered': "there'd", "thered've": "there'd've", "there'dve": "there'd've", 'therere': "there're", 'theres': "there's", 'theyd': "they'd", "theyd've": "they'd've", "they'dve": "they'd've", 'theyll': "they'll", 'theyre': "they're", 'theyve': "they've", 'twas': "'twas", 'wasnt': "wasn't", "wed've": "we'd've", "we'dve": "we'd've", 'weve': "we've", 'werent': "weren't", 'whatll': "what'll", 'whatre': "what're", 'whats': "what's", 'whatve': "what've", 'whens': "when's", 'whered': "where'd", 'wheres': "where's", 'whereve': "where've", 'whod': "who'd", "whod've": "who'd've", "who'dve": "who'd've", 'wholl': "who'll", 'whos': "who's", 'whove': "who've", 'whyll': "why'll", 'whyre': "why're", 'whys': "why's", 'wont': "won't", 'wouldve': "would've", 'wouldnt': "wouldn't", "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've", 'yall': "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", 'youd': "you'd", "youd've": "you'd've", "you'dve": "you'd've", 'youll': "you'll", 'youre': "you're", 'youve': "you've"}
        self.manualMap = {'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'}
        self.articles = ['a', 'an', 'the']
        self.periodStrip = re.compile('(?!<=\\d)(\\.)(?!\\d)')
        self.commaStrip = re.compile('(\\d)(\\,)(\\d)')
        self.punct = [';', '/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + ' ' in inText or ' ' + p in inText) or re.search(self.commaStrip, inText) != None:
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = self.periodStrip.sub('', outText, re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for (wordId, word) in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = ' '.join(outText)
        return outText

@lru_cache(maxsize=32768)
def normalize_answer(text):
    evaluator = _VQANormalizer()
    text = text.replace("\n", " ").replace("\t", " ").strip()
    return evaluator.processDigitArticle(evaluator.processPunctuation(text))


def independent_vqa(output, row):
    prediction = normalize_answer(output)
    answers = [normalize_answer(a["answer"]) for a in row["answers"]]
    return sum(min(1, sum(prediction == a for j, a in enumerate(answers) if j != i) / 3)
               for i in range(len(answers))) / len(answers)

def gqa_normalize(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation))

def chartqa_correct(prediction, target):
    def numeric(text):
        try:
            return float(text.rstrip('%'))/100 if text.endswith('%') else float(text)
        except ValueError:
            return None
    p, t = numeric(prediction), numeric(target)
    if p is not None and t:
        return abs(p-t)/abs(t) <= .05
    return prediction.lower() == target.lower()

def edit_distance(a,b):
    if a==b:
        return 0
    if len(a)<len(b):
        a,b=b,a
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]

def anls(prediction,answers):
    prediction=prediction.lower().strip()
    best=0.0
    for answer in answers:
        answer=answer.lower().strip()
        length=max(len(prediction),len(answer))
        distance=edit_distance(prediction,answer)/length if length else 0
        best=max(best,1-distance if distance<.5 else 0)
    return best

def choice_prediction(output,choices):
    stripped=output.strip().strip(' .!()[]').upper()
    if stripped in choices:
        return stripped
    normalized=normalize_answer(output)
    matches=[k for k,v in choices.items() if len(v.strip())>1 and normalize_answer(v)==normalized]
    return matches[0] if len(matches)==1 else None

def binary_prediction(output):
    match=re.match(r'^\W*(yes|no|true|false)\b',output.strip(),re.I)
    if not match:
        return None
    return 'yes' if match.group(1).lower() in ('yes','true') else 'no'

def pope_prediction(output):
    words=output.split('.')[0].replace(',','').split(' ')
    return 'no' if any(w in words for w in ('No','not','no')) else 'yes'

def check_is_number(string):
    try:
        float(string.replace(',', ''))
        return True
    except ValueError:
        return False

def normalize_str(string):

    string = string.strip()

    is_number = check_is_number(string)

    if is_number:
        string = string.replace(',', '')
        string = float(string)
        string = round(string, 2)
        return [string]
    else:
        string = string.lower()
        if len(string) == 1:
            return [" " + string, string + " "]
        return [string]

def extract_numbers(string):
    pattern_commas = r'-?\b\d{1,3}(?:,\d{3})+\b'
    pattern_scientific = r'-?\d+(?:\.\d+)?[eE][+-]?\d+'
    pattern_simple = r'-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])'

    numbers_with_commas = re.findall(pattern_commas, string)
    numbers_scientific = re.findall(pattern_scientific, string)
    numbers_simple = re.findall(pattern_simple, string)

    all_numbers = numbers_with_commas + numbers_scientific + numbers_simple
    return all_numbers

def parse_open_response(response):
    def get_key_subresponses(response):
        key_responses = []
        response = response.strip().strip(".").lower()
        sub_responses = re.split(r'\.\s(?=[A-Z])|\n', response)
        indicators_of_keys = ['could be ', 'so ', 'is ',
                            'thus ', 'therefore ', 'final ', 'answer ', 'result ']
        key_responses = []
        for index, resp in enumerate(sub_responses):
            if index == len(sub_responses) - 1:
                indicators_of_keys.extend(['='])
            shortest_key_response = None
            for indicator in indicators_of_keys:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(indicator)[-1].strip()
                    else:
                        if len(resp.split(indicator)[-1].strip()) < len(shortest_key_response):
                            shortest_key_response = resp.split(indicator)[-1].strip()

            if shortest_key_response:
                if shortest_key_response.strip() not in [":", ",", ".", "!", "?", ";", ":", "'"]:
                    key_responses.append(shortest_key_response)
        if len(key_responses) == 0:
            return [response]
        return key_responses
    key_responses = get_key_subresponses(response)

    pred_list = key_responses.copy()
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))

    tmp_pred_list = []
    for i in range(len(pred_list)):
        tmp_pred_list.extend(normalize_str(pred_list[i]))
    pred_list = tmp_pred_list

    pred_list = list(set(pred_list))

    return pred_list

def eval_open(gold_i, pred_i):
    correct = False
    if isinstance(gold_i, list):
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(normalize_str(answer))
    else:
        norm_answers = normalize_str(gold_i)
    for pred in pred_i:
        if isinstance(pred, str):
            for norm_ans in norm_answers:
                if isinstance(norm_ans, str) and norm_ans in pred:
                    if not correct:
                        correct = True
                    break
        else:
            if pred in norm_answers:
                if not correct:
                    correct = True
                break
    return correct

def clustered(values,ids):
    x=np.asarray(values,dtype=float);mean=float(x.mean());n=len(x)
    groups=sorted(set(ids));m=len(groups)
    residuals=np.array([sum(v-mean for v,i in zip(x,ids) if i==g) for g in groups])
    se=float(np.sqrt(m/(m-1)*np.square(residuals).sum()/n**2))
    return {'score':100*mean,'se':100*se,'n':n,'image_clusters':m}

def uncertainty(values, ids):
    import numpy as np
    groups = sorted(set(ids))
    if len(groups) < 2:
        return {'score': 100*sum(values)/len(values), 'n': len(values),
                'image_clusters': len(groups), 'se': None, 'ci95': None}
    summary = clustered(values, ids)
    sums = np.array([sum(v for v, i in zip(values, ids) if i == g) for g in groups])
    counts = np.array([ids.count(g) for g in groups])
    sample = np.random.default_rng(42).integers(0, len(groups), (10000, len(groups)))
    summary['ci95'] = np.percentile(100*sums[sample].sum(1)/counts[sample].sum(1), [2.5, 97.5]).tolist()
    return summary

def cluster_ids(records, captions):
    parents = {}
    def find(x):
        parents.setdefault(x, x)
        if parents[x] != x:
            parents[x] = find(parents[x])
        return parents[x]
    bundles = [captions[r['image_sha256']].get('image_sha256s', [r['image_sha256']]) for r in records]
    for hashes in bundles:
        for h in hashes:
            parents[find(h)] = find(hashes[0])
    return [find(hashes[0]) for hashes in bundles]

def score_answer(output, row):
    metric = row['metric']
    answers = [a['answer'] for a in row['answers']]
    if metric == 'vqa':
        return independent_vqa(output, row)
    if metric == 'gqa_exact':
        return float(gqa_normalize(output) == gqa_normalize(answers[0]))
    if metric == 'chartqa_relaxed':
        return float(chartqa_correct(output, answers[0]))
    if metric == 'anls':
        return anls(output, answers)
    if metric == 'mcq':
        if 'choices' in row:
            prediction = choice_prediction(output, row['choices'])
        else:
            match = re.match(r'^\s*[([]?\s*([A-D])\b', output.upper())
            prediction = match.group(1) if match else None
        return float(prediction == row['correct_choice'])
    if metric == 'aokvqa_da':
        prediction = normalize_answer(output)
        return min(1.0, sum(prediction == normalize_answer(a) for a in answers) / 3)
    if metric == 'normalized_exact':
        return float(any(normalize_answer(output) == normalize_answer(a) for a in answers))
    if metric == 'binary_prefix':
        return float(binary_prediction(output) == answers[0])
    if metric == 'pope_accuracy':
        return float(pope_prediction(output) == answers[0])
    if metric == 'mmmu_open_accuracy':
        return float(eval_open(answers, parse_open_response(output)))
    if metric == 'mme_accuracy':
        return float(mme_prediction(output) == answers[0].lower())
    if metric == 'mathvision_accuracy':
        answer = mathvision_answer(output)
        return float(any(mathvision_equal(str(gt), answer) for gt in
                         (answers[0], row.get('choices', {}).get(answers[0], ''))))
    raise ValueError(f'Unknown metric: {metric}')


@lru_cache(maxsize=32768)
def mathvision_number(text):
    from latex2sympy2_extended import latex2sympy
    if '\\!' in text:
        raise ValueError('Unsupported reference LaTeX spacing command')
    text = text.replace('$', ' ')
    return round(float(latex2sympy(text, normalization_config=None).evalf()), 2)


def mathvision_tuple(text):
    parts = text[1:-1].split(',')
    if (text.startswith('(') and text.endswith(')') or text.startswith('[') and text.endswith(']')) and len(parts) > 1:
        try:
            parts = [str(mathvision_number(p)) if 'infty' not in p and p not in ('a', '-a') else p for p in parts]
            return text[0] + ','.join(parts) + text[-1]
        except Exception:
            pass
    return text


@lru_cache(maxsize=32768)
def mathvision_equal(left, right):
    left, right = left.lower(), right.lower()
    if not left.replace(' ', '') or not right.replace(' ', ''):
        return False
    if left.strip() == right.strip():
        return True
    left, right = mathvision_tuple(left), mathvision_tuple(right)
    if left == right:
        return True
    try:
        return mathvision_number(left) == mathvision_number(right)
    except Exception:
        return False


@lru_cache(maxsize=32768)
def mathvision_answer(output):
    answer = output.strip()
    for c in 'ABCDE':
        if (answer.endswith(f' {c}.') or answer.endswith(f' ({c}).') or answer.startswith(f'{c}\n')
                or answer.startswith(f'({c})\n') or answer.startswith(f'({c}) {c}\n')):
            answer = c
    try:
        float(answer.split('is ')[-1].rstrip('.'))
        answer = answer.split('is ')[-1].rstrip('.')
    except ValueError:
        pass
    if 'oxed{' not in answer:
        for flag in ('the final answer is', 'the answer is', 'the correct answer is', 'the answer should be'):
            for phrase in (flag, flag.replace('the', 'The')):
                before = answer
                answer = answer.split(phrase)[-1].strip()
                if phrase in before:
                    answer = answer.split('\n')[0].split('. ')[0]
    elif answer.count('oxed{') > 1:
        answer = '\\boxed{' + answer.split('oxed{')[-1]
    answer = mathvision_find_math_answer(answer)
    for c in 'abcde':
        answer = answer.replace(f'({c})', c).replace('{' + c + '}', c)
    return answer.rstrip('.').lstrip(':').strip()


def mathvision_fix_fracs(string):
    substrs = string.split('\\frac')
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += '\\frac'
            if len(substr) > 0 and substr[0] == '{':
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != '{':
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}{' + b + '}' + post_substr
                    else:
                        new_str += '{' + a + '}{' + b + '}'
                elif len(substr) > 2:
                    post_substr = substr[2:]
                    new_str += '{' + a + '}' + b + post_substr
                else:
                    new_str += '{' + a + '}' + b
    string = new_str
    return string

def mathvision_fix_a_slash_b(string):
    if len(string.split('/')) != 2:
        return string
    (a, b) = string.split('/')
    try:
        a = int(a)
        b = int(b)
        assert string == '{}/{}'.format(a, b)
        new_string = '\\frac{' + str(a) + '}{' + str(b) + '}'
        return new_string
    except:
        return string

def mathvision_remove_right_units(string):
    splits = string.split('\\text{ ')
    return splits[0]

def mathvision_fix_sqrt(string):
    if '\\sqrt' not in string:
        return string
    splits = string.split('\\sqrt')
    new_string = splits[0]
    for split in splits[1:]:
        if len(split) > 0 and split[0] != '{':
            a = split[0]
            new_substr = '\\sqrt{' + a + '}' + split[1:]
        else:
            new_substr = '\\sqrt' + split
        new_string += new_substr
    return new_string

def mathvision_strip_string(string):
    string = string.replace('\n', '')
    string = string.replace('\\!', '')
    string = string.replace('\\\\', '\\')
    string = string.replace('tfrac', 'frac')
    string = string.replace('dfrac', 'frac')
    string = string.replace('\\left', '')
    string = string.replace('\\right', '')
    string = string.replace('^{\\circ}', '')
    string = string.replace('^\\circ', '')
    string = string.replace('\\$', '')
    string = string.replace('$', '')
    string = mathvision_remove_right_units(string)
    string = string.replace('\\%', '')
    string = string.replace('\\%', '')
    string = string.replace(' .', ' 0.')
    string = string.replace('{.', '{0.')
    if len(string) == 0:
        return string
    if string[0] == '.':
        string = '0' + string
    if len(string.split('=')) == 2:
        string = string.split('=')[-1]
    if len(string.split('\\approx')) == 2:
        string = string.split('\\approx')[-1]
    if 'sqrt' in string:
        string = mathvision_fix_sqrt(string)
    string = string.replace(' ', '')
    if 'sqrt' in string:
        string = mathvision_fix_fracs(string)
    if string == '0.5':
        string = '\\frac{1}{2}'
    string = mathvision_fix_a_slash_b(string)
    return string

def mathvision_find_math_answer(s: str) -> str:
    s = s.lower()
    if '{}' in s:
        s = s.replace('{}', '')
    try:
        pattern = re.compile('oxed{(.*)}', flags=re.S)
        ans = pattern.findall(s)[-1]
    except:
        ans = s
    if ans.find('}') != -1 and (ans.find('{') == -1 or ans.find('}') < ans.find('{')):
        ans = ans.split('}')[0]
    ans = ans.split('=')[-1]
    ans = ans.split('\\approx')[-1]
    ans = ans.replace(' ', '').replace('\\,', '').replace('∞', '\\infty')
    ans = ans.replace('+\\infty', '\\infty').replace('\\\\', '\\').replace('\n', '')
    ans = ans.replace('\\text', '').replace('\\mbox', '').replace('bmatrix', 'pmatrix')
    ans = ans.replace('\\left', '').replace('\\right', '').replace('^{\\circ}', '')
    ans = ans.replace('^\\circ', '').replace('{m}^3', '').replace('m^3', '')
    ans = ans.replace('{units}', '').replace('units', '').replace('{km}', '').replace('km', '')
    return mathvision_strip_string(ans)
