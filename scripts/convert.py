import argparse
import json
from pathlib import Path
from typing import Optional

from static import *


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        type=str,
        default=None,
        help="Input JSON path. If omitted, use json/{CORPUS_NAME}_positive.json.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output JSON path. If omitted, derive it from the input JSON path.",
    )
    return parser.parse_args()

def delete_negation(marked_words: list[dict]) -> list[dict]:
    new_marked_words = []
    for word_info in marked_words:
        if word_info['mark'] == 'negation':
            continue
        new_marked_words.append(word_info)
    return new_marked_words


def santangen(word: str) -> str:
    if word.isupper():
        word_form = 'upper'
    elif word.istitle():
        word_form = 'title'
    else:
        word_form = 'lower'
    word = word.lower()
    if word.endswith(('s', 'sh', 'ch', 'x', 'o')):
        res = word + 'es'
    elif word.endswith('y') and word[-2] not in 'aeiou':
        res = word[:-1] + 'ies'
    else:
        res = word + 's'
    if word_form == 'upper':
        res = res.upper()
    elif word_form == 'title':
        res = res.title()
    return res

def past_form(word: str) -> str:
    if word.isupper():
        word_form = 'upper'
    elif word.istitle():
        word_form = 'title'
    else:
        word_form = 'lower'
    word = word.lower()
    if word in PAST_FORM_MAP:
        res = PAST_FORM_MAP[word]
    elif word.endswith('e'):
        res = word + 'd'
    elif word.endswith('y') and word[-2] not in 'aeiou':
        res = word[:-1] + 'ied'
    elif word[-2] in 'aeiou' and word[-1] not in 'aeiou':
        res = word + word[-1] + 'ed'
    else:
        res = word + 'ed'
    if word_form == 'upper':
        res = res.upper()
    elif word_form == 'title':
        res = res.title()
    return res

def fix_auxiliary(marked_words: list[dict]) -> list[dict]:
    verb_idx = [i for i, word in enumerate(marked_words) if word['mark'] == 'verb'][0]
    verb_auxiliary_idxes = [i 
                          for i, word in enumerate(marked_words) if word['mark'] == 'auxiliary' 
                          and word['word'].lower() in VERB_AUXILIARY]
    if verb_auxiliary_idxes != []:
        verb_auxiliary_idx = verb_auxiliary_idxes[0]
    else:
        verb_auxiliary_idx = None
    
    for i, word in enumerate(marked_words):
        if word['mark'] == 'auxiliary' and word['word'].lower() == 'does':
            if verb_auxiliary_idx is not None:
                marked_words[verb_auxiliary_idx]['word'] = santangen(marked_words[verb_auxiliary_idx]['word'])
            else:
                marked_words[verb_idx]['word'] = santangen(marked_words[verb_idx]['word'])
            marked_words.pop(i)
            break
        elif word['mark'] == 'auxiliary' and word['word'].lower() == 'did':
            if verb_auxiliary_idx is not None:
                marked_words[verb_auxiliary_idx]['word'] = past_form(marked_words[verb_auxiliary_idx]['word'])
            else:   
                marked_words[verb_idx]['word'] = past_form(marked_words[verb_idx]['word'])
            marked_words.pop(i)
            break
        elif word['mark'] == 'auxiliary' and word['word'].lower() == 'do':
            marked_words.pop(i)
            break
        elif word['mark'] == 'auxiliary' and word['word'].lower() == 'ca':
            marked_words[i]['word'] = 'can'
            break
        elif word['mark'] == 'auxiliary' and word['word'].lower() == 'wo':
            marked_words[i]['word'] = 'will'
            break
        
    return marked_words

def fix_negative_oriented_words(marked_words: list[dict]) -> list[dict]:
    new_marked_words = []
    for i, word_info in enumerate(marked_words):
        if word_info['word'].isupper():
                word_form = 'upper'
        elif word_info['word'].istitle():
            word_form = 'title'
        else:
            word_form = 'lower'
        if word_info['word'].lower() in NEGATIVE_POSITIVE_ONE_WORD_MAP:
            new_word = NEGATIVE_POSITIVE_ONE_WORD_MAP[word_info['word'].lower()]
        elif i > 0 and marked_words[i-1]['word'].lower() + ' ' + word_info['word'].lower() in NEGATIVE_POSITIVE_TWO_WORDS_MAP:
            new_word = NEGATIVE_POSITIVE_TWO_WORDS_MAP[marked_words[i-1]['word'].lower() + ' ' + word_info['word'].lower()]
            new_marked_words.pop()
        else:
            new_word = word_info['word']
        if word_form == 'upper':
            new_word = new_word.upper()
        elif word_form == 'title':
            new_word = new_word.title()
        new_marked_words.append({'word': new_word, 'mark': word_info['mark'], 'named_entity': word_info['named_entity']})
        
    return new_marked_words

def is_start_of_sentence(word_infos: list[dict]) -> bool:
    return word_infos[0]['word'].istitle() and word_infos[0]['named_entity'] == 'O'

def replace_focus(marked_words: list[dict], semantic_role: str, named_entity: str) -> list[dict]:
    new_marked_words = []
    focus_words = [word_info for word_info in marked_words if word_info['mark'] == 'focus']
    for i, marked_word in enumerate(marked_words):
        if not marked_word["word"].isalpha():
            continue
        elif marked_word["mark"] == "focus" and marked_word["word"].istitle():
            start_of_sentence = True
            break
        else:
            start_of_sentence = False
            break

    if semantic_role in ['A0', 'A1', 'A2', 'A3', 'A4'] and (named_entity == 'PERSON' or all(focus_word['word'].lower() in ['i', 'you', 'he', 'she'] for focus_word in focus_words)):
        replacement_word = [{'word': 'someone', 'mark': 'replacement', 'named_entity': 'O'}]
    elif semantic_role in ['A0', 'A1', 'A2', 'A3', 'A4']:
        replacement_word = [{'word': 'something', 'mark': 'replacement', 'named_entity': 'O'}]
    elif semantic_role in ['A0', 'A1', 'A2', 'A3', 'A4'] and all(focus_word['word'].lower() in ['they', 'we'] for focus_word in focus_words):
        replacement_word = [{'word': 'some', 'mark': 'replacement', 'named_entity': 'O'}, {'word': 'people', 'mark': 'replacement', 'named_entity': 'O'}]
    else:
        replacement_word = SEMANTIC_ROLE_MAP[semantic_role]
    if start_of_sentence:
        replacement_word = [{'word': word_info['word'].title(), 'mark': word_info['mark'], 'named_entity': word_info['named_entity']} 
                            if i == 0 else word_info 
                            for i, word_info in enumerate(replacement_word)]
    focus_detection = False
    for word_info in marked_words:
        if word_info['mark'] == 'focus':
            if not focus_detection:
                focus_detection = True
                new_marked_words += replacement_word
            else:
                continue
        else:
            new_marked_words.append(word_info)
    if start_of_sentence and (not (len(focus_words) == 1 and focus_words[0]['word'] == 'I') and focus_words[0]['named_entity'] == 'O'):
        focus_words = [{'word': word_info['word'].lower(), 'mark': word_info['mark'], 'named_entity': word_info['named_entity']} 
                       if i == 0 else word_info 
                       for i, word_info in enumerate(focus_words)]
    insert_idx = new_marked_words.index({'word': '', 'mark': 'end_of_focus', 'named_entity': 'O'})
    if insert_idx > 0 and new_marked_words[insert_idx - 1]['word'][-1] in ['.', '!', '?']:
        insert_idx -= 1
    if insert_idx < len(new_marked_words) - 1 and any(char.isalpha() for char in new_marked_words[insert_idx + 1]['word']):
        insert_words = [{'word': ',', 'mark': 'other', 'named_entity': 'O'}, {'word': 'but', 'mark': 'other', 'named_entity': 'O'}, {'word': 'not', 'mark': 'other', 'named_entity': 'O'}] + focus_words + [{'word': ',', 'mark': 'other', 'named_entity': 'O'}]
        
    else:
        insert_words = [{'word': ',', 'mark': 'other', 'named_entity': 'O'}, {'word': 'but', 'mark': 'other', 'named_entity': 'O'}, {'word': 'not', 'mark': 'other', 'named_entity': 'O'}] + focus_words
    new_marked_words.pop(new_marked_words.index({'word': '', 'mark': 'end_of_focus', 'named_entity': 'O'}))
    new_marked_words[insert_idx:insert_idx] = insert_words
    return new_marked_words

def convertable_sentence(marked_words: list[dict], senmantic_role: str) -> bool:
    negation_idx = [i for i, word in enumerate(marked_words) if word['mark'] == 'negation']
    negation_cues = [word['word'] for word in marked_words if word['mark'] == 'negation']
    words = [word['word'] for word in marked_words]
    auxiliary_words = [word['word'] for word in marked_words if word['mark'] == 'auxiliary']
    if len(negation_idx) == 0:
        return False
    return senmantic_role in CONVERATABLE_SEMANTIC_ROLES and all(negation_cue.lower() in NEGATION_CUES for negation_cue in negation_cues) and (all(word.lower() in AUXILIARY_VERBS for word in auxiliary_words) or all(i < len(words) - 1 and words[i + 1].lower() == 'to' for i in negation_idx))

def default_paths(input_json: Optional[str] = None, output_path: Optional[str] = None) -> tuple[Path, Path]:
    if input_json is None:
        json_path = Path(f'json/{CORPUS_NAME}_positive.json')
        default_output_path = Path(f'dataset/converted/{CORPUS_NAME}_converted.json')
    else:
        json_path = Path(input_json)
        input_stem = json_path.stem
        if input_stem.endswith("_positive"):
            input_stem = input_stem[:-len("_positive")]
        default_output_path = Path("dataset") / "converted" / f"{input_stem}_converted.json"

    return json_path, Path(output_path) if output_path is not None else default_output_path


def convert(input_json: Optional[str] = None, output_path: Optional[str] = None):
    json_path, output_path = default_paths(input_json, output_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        semantic_role_infos = json.load(f)
    sentence_infos = semantic_role_infos['infos']
    sentences = []
    instance_id = 0
    for i, sentence_info in enumerate(sentence_infos):
        sentence_id = sentence_info["id"]
        marked_words = sentence_info['sentence']
        semantic_role = sentence_info['semantic_role_of_focus']
        named_entity = sentence_info['focus_named_entity']
        previous_sentence = sentence_info['previous_sentence']
        subsequent_sentence = sentence_info['subsequent_sentence']
        before_sentence = ' '.join([marked_word['word'] for marked_word in marked_words])
        if not convertable_sentence(marked_words, semantic_role):
            print(f"sentence {sentence_id} is not convertable.")
            continue
        marked_words = delete_negation(marked_words)
        marked_words = fix_auxiliary(marked_words)
        marked_words = fix_negative_oriented_words(marked_words)
        marked_words = replace_focus(marked_words, semantic_role, named_entity)
        sentences.append({"sentence_id": sentence_id, 
                          "instance_id": instance_id, 
                          "previous_sentence": previous_sentence, 
                          "subsequent_sentence": subsequent_sentence, 
                          "sentence": ' '.join([word_info['word'] for word_info in marked_words]), 
                          "before_sentence": before_sentence, 
                          "focus_words": ' '.join([word_info['word'] for word_info in marked_words if word_info['mark'] == 'focus']),
                          "replacement_words": ' '.join([word_info['word'] for word_info in marked_words if word_info['mark'] == 'replacement']),
                          })
        instance_id += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sentences, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    args = make_args()
    convert(input_json=args.input_json, output_path=args.output_path)
