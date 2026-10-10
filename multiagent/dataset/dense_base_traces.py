import re
from multiagent.create_phase2_dataset import arc_choices_text

DEMO = {
    'gsm8k': 'Problem: A shop sells 8 pens on Monday and twice as many on Tuesday. How many pens were sold altogether?\nCorrect answer: 24\nSolution: On Tuesday the shop sold 2 * 8 = 16 pens. The total is 8 + 16 = 24 pens.\nAnswer: 24',
    'math-algebra': 'Problem: Solve 3x + 2 = 14 for x.\nCorrect answer: 4\nSolution: Subtract 2 from both sides to get 3x = 12. Dividing both sides by 3 gives x = 4.\nAnswer: \\boxed{4}',
    'arc-c': 'Problem: Which material conducts electricity best?\nA. Rubber\nB. Copper\nC. Glass\nD. Wood\nCorrect answer: B\nSolution: Copper is a metal with mobile electrons that carry electric current. Rubber, glass and dry wood are insulators. Therefore copper is the best conductor among these choices.\nAnswer: B',
}
STOP_STRINGS = ['\nProblem:', '\nQuestion:']


def prompt(task, sample):
    question = sample['question']
    if task == 'arc-c':
        question += '\n' + arc_choices_text(sample)
    return DEMO[task] + f"\n\nProblem: {question}\nCorrect answer: {sample['answer']}\nSolution:"


def extract(raw):
    return re.split(r'\n\s*(?:Problem|Question):', raw, maxsplit=1)[0].strip()
