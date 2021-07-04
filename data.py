import json
import pathlib
import secrets
import typing

import torch
import torch.utils.data
import transformers


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        problems: typing.Union[dict[str, dict], list[dict]],
        tokenizer: transformers.tokenization_utils_fast.PreTrainedTokenizerFast,
        title: bool = True,
        related: bool = True,
        question: bool = True,
        hint: bool = True,
        author: bool = True,
        the_rest: bool = False,
    ) -> None:
        super().__init__()
        self.problems = problems
        self.tokenizer = tokenizer
        text = []
        answer = []
        sep = '。'
        for problem in problems:
            if isinstance(problem, str):
                problem = problems[problem]
            t = ''
            if title is True:
                t += problem['title'].strip()
                t += sep
            if related is True:
                t += '關聯：{}'.format(problem['related'].strip())
                t += sep
            if question is True:
                t += '問題：{}'.format(problem['question'].strip())
            start, end = 0, 0
            for i, option in enumerate(problem['options']):
                option = option.strip()
                t += '{}.'.format(i + 1)
                if i == problem['answer']:
                    start = len(t)
                    end = start + len(option) - 1
                t += option
                t += sep
            answer.append([start, end])
            if hint is True:
                t += '小提示：{}'.format(problem['hint'].strip())
                t += sep
            if author is True:
                t += '出題者：{}'.format(problem['author'].strip())
                t += sep
            if the_rest is True:
                t += problem['the_rest'].strip()
            text.append(t)
        self.text = text
        self.tokenized_text = tokenizer(text, add_special_tokens=True, padding=True)
        self.answer = answer

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, index):
        encoding = self.tokenized_text[index]
        start = encoding.char_to_token(self.answer[index][0])
        end = encoding.char_to_token(self.answer[index][1])
        if start is None or end is None:
            raise ValueError
        return (
            torch.tensor(encoding.ids),
            torch.tensor(encoding.type_ids),
            torch.tensor(encoding.attention_mask),
            start,
            end,
            index,
        )


class MaskedLMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sentences: list[str],
        tokenizer: transformers.tokenization_utils_fast.PreTrainedTokenizerFast,
        max_length: int = 200,
        stride: int = 75,
        mask_prob: float = 0.15,
    ) -> None:
        super().__init__()
        self.sentences = sentences
        self.tokenizer = tokenizer
        tokenized_sentences = tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_tensors='pt',
            return_overflowing_tokens=True,
        )
        self.tokenized_sentences = tokenized_sentences
        self.labels: torch.Tensor = tokenized_sentences.input_ids.detach().clone()
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        encoding = self.tokenized_sentences[index]
        ids = torch.tensor(encoding.ids)
        # type_ids = torch.tensor(encoding.type_ids)
        attention_mask = torch.tensor(encoding.attention_mask)
        labels = self.labels[index]

        rand = torch.rand(ids.shape)
        mask = (rand < self.mask_prob)
        for name in self.tokenizer.special_tokens_map:
            if name != 'unk_token':
                mask &= ids != getattr(self.tokenizer, '{}_id'.format(name))
        ids[mask] = self.tokenizer.mask_token_id

        return ids, attention_mask, labels


def split(problems: typing.Union[dict[str, dict], list[dict]], proportion: float = 0.2, shuffle: bool = True):
    if isinstance(problems, dict):
        x = list(problems.keys())
    else:
        x = list(range(len(problems)))

    if shuffle is True:
        system_random = secrets.SystemRandom()
        system_random.shuffle(x)
    x = x[::-1]
    sep = int(len(x) * proportion)
    b = x[:sep][::-1]
    a = x[sep:][::-1]
    if isinstance(problems, dict):
        a = {key: problems[key] for key in a}
        b = {key: problems[key] for key in b}
    else:
        a = [problems[index] for index in a]
        b = [problems[index] for index in b]
    return a, b


def split_from_path(
    problems: typing.Union[str, pathlib.Path],
    a: typing.Union[str, pathlib.Path] = None,
    b: typing.Union[str, pathlib.Path] = None,
    proportion: float = 0.2,
    shuffle: bool = True,
):
    problems = pathlib.Path(problems)
    stem = problems.stem
    if a is None:
        a = pathlib.Path('{}_a.json'.format(stem))
        assert not a.exists()
    else:
        a = pathlib.Path(a)
    if b is None:
        b = pathlib.Path('{}_b.json'.format(stem))
        assert not b.exists()
    else:
        b = pathlib.Path(b)
    problems = json.loads(problems.read_text())
    A, B = split(problems, proportion, shuffle)
    with a.open('w') as fp:
        json.dump(A, fp, ensure_ascii=False)
    with b.open('w') as fp:
        json.dump(B, fp, ensure_ascii=False)
