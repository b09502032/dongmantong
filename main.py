import csv
import json
import pathlib
import shutil
import typing

import torch
import torch.utils.data
import transformers
import transformers.models
import transformers.modeling_outputs
import tqdm

import data
import utils


def main(
    train: typing.Union[str, pathlib.Path],
    validation: typing.Union[str, pathlib.Path] = None,
    test: typing.Union[str, pathlib.Path] = None,
    pretrained_model_name: str = 'hfl/chinese-macbert-large',
    log: typing.Union[str, pathlib.Path] = None,
    num_epochs: int = 100,
    batch_size: int = 16,
    lr: float = 5e-6,
    seed: int = 713574848,
    fp16: bool = False,
    device: torch.device = None,
):
    info_path = 'info.json'
    loss_plot_path = 'loss.png'
    accuracy_plot_path = 'accuracy.png'
    validation_prediction_path = 'validation_prediction.csv'

    info = {
        'pretrained_model_name': pretrained_model_name,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'lr': lr,
        'seed': seed,
        'fp16': fp16,
        'train': {
            'loss': [],
            'accuracy': [],
        },
    }

    utils.same_seeds(seed)

    if fp16 is True:
        import accelerate
        accelerator = accelerate.Accelerator(fp16=True)
        device = accelerator.device
    else:
        if device is None:
            device = utils.get_device()

    model = transformers.BertForQuestionAnswering.from_pretrained(pretrained_model_name)
    model: transformers.modeling_utils.PreTrainedModel
    model = model.to(device)

    tokenizer = transformers.BertTokenizerFast.from_pretrained('hfl/chinese-macbert-large')
    tokenizer: transformers.tokenization_utils_fast.PreTrainedTokenizerFast

    train = pathlib.Path(train)
    train_problems = json.loads(train.read_text())
    train_dataset = data.Dataset(train_problems, tokenizer)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, True, pin_memory=True)

    validation_loader = None
    if validation is not None:
        validation = pathlib.Path(validation)
        validation_problems = json.loads(validation.read_text())
        validation_dataset = data.Dataset(validation_problems, tokenizer)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size, pin_memory=True)

        info['validation'] = {'loss': [], 'accuracy': [], 'prediction': []}

    optimzer = transformers.AdamW(model.parameters(), lr=lr)

    if fp16 is True:
        model, optimzer, train_loader = accelerator.prepare(model, optimzer, train_loader)

    if log is not None:
        import matplotlib.pyplot
        import matplotlib.ticker
        log = pathlib.Path(log)
        if log.is_dir():
            shutil.rmtree(log)
        log.mkdir()
        info_path = log / info_path
        loss_plot_path = log / loss_plot_path
        accuracy_plot_path = log / accuracy_plot_path
        validation_prediction_path = log / validation_prediction_path

    for epoch in tqdm.tqdm(range(num_epochs)):
        model.train()
        train_loss = train_acc = 0
        for batch in (train_loader):
            batch: list[torch.Tensor]
            batch[:5] = [_.to(device) for _ in batch[:5]]
            output: transformers.modeling_outputs.QuestionAnsweringModelOutput = model(
                input_ids=batch[0],
                token_type_ids=batch[1],
                attention_mask=batch[2],
                start_positions=batch[3],
                end_positions=batch[4],
            )
            start_index = torch.argmax(output.start_logits, dim=1)
            end_index = torch.argmax(output.end_logits, dim=1)
            train_acc += ((start_index == batch[3]) & (end_index == batch[4])).sum()
            train_loss += output.loss * batch[4].shape[0]

            if fp16 is True:
                accelerator.backward(output.loss)
            else:
                output.loss.backward()
            optimzer.step()
            optimzer.zero_grad()
        train_loss = train_loss.item() / len(train_loader.dataset)
        train_acc = train_acc.item() / len(train_loader.dataset)
        info['train']['loss'].append(train_loss)
        info['train']['accuracy'].append(train_acc)
        print('Train | Epoch {} | loss = {:.5f}, acc = {:.5f}'.format(epoch + 1, train_loss, train_acc))
        if validation_loader is not None:
            model.eval()
            validation_loss = validation_acc = 0
            validation_prediction = []
            with torch.no_grad():
                for batch in (validation_loader):
                    batch: list[torch.Tensor]
                    batch[:5] = [_.to(device) for _ in batch[:5]]
                    output: transformers.modeling_outputs.QuestionAnsweringModelOutput = model(
                        input_ids=batch[0],
                        token_type_ids=batch[1],
                        attention_mask=batch[2],
                        start_positions=batch[3],
                        end_positions=batch[4],
                    )
                    start_index = torch.argmax(output.start_logits, dim=1)
                    end_index = torch.argmax(output.end_logits, dim=1)
                    validation_loss += output.loss * batch[4].shape[0]
                    validation_acc += ((start_index == batch[3]) & (end_index == batch[4])).sum()
                    for i, index in enumerate(batch[5]):
                        index = index.item()
                        text = validation_dataset.text[index]
                        choice = ''
                        start = validation_dataset.tokenized_text[index].token_to_chars(start_index[i])
                        stop = validation_dataset.tokenized_text[index].token_to_chars(end_index[i])
                        if start is not None and stop is not None:
                            start = start[0]
                            stop = stop[1]
                            choice = text[start:stop]
                        answer = validation_dataset.answer[index]
                        p = [index, text, text[answer[0]:answer[1] + 1], choice]
                        validation_prediction.append(p)
            validation_loss = validation_loss.item() / len(validation_loader.dataset)
            validation_acc = validation_acc.item() / len(validation_loader.dataset)
            validation_prediction.sort(key=lambda x: x[2] == x[3], reverse=True)
            info['validation']['loss'].append(validation_loss)
            info['validation']['accuracy'].append(validation_acc)
            info['validation']['prediction'].append(validation_prediction)
            if log is not None:
                with validation_prediction_path.open('w') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['index', 'text', 'answer', 'prediction'])
                    writer.writerows(validation_prediction)

            print('Validation | Epoch {} | loss = {:.5f}, acc = {:.5f}'.format(epoch + 1, validation_loss, validation_acc))
        if log is not None:
            info_path.write_text(json.dumps(info, ensure_ascii=False, separators=(',', ':')))
            fig, ax = matplotlib.pyplot.subplots()
            ax.plot(range(epoch + 1), info['train']['loss'], label='train')
            if 'validation' in info:
                ax.plot(range(epoch + 1), info['validation']['loss'], label='validation')
            ax.set_title('loss')
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')
            ax.legend()
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            fig.savefig(loss_plot_path)
            matplotlib.pyplot.close(fig)
            fig, ax = matplotlib.pyplot.subplots()
            ax.plot(range(epoch + 1), info['train']['accuracy'], label='train')
            if 'validation' in info:
                ax.plot(range(epoch + 1), info['validation']['accuracy'], label='validation')
            ax.set_title('accuracy')
            ax.set_xlabel('epoch')
            ax.set_ylabel('accuracy')
            ax.legend()
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            fig.savefig(accuracy_plot_path)
            matplotlib.pyplot.close(fig)


if __name__ == '__main__':
    import argparse
    import inspect
    signature = inspect.signature(main)
    default = {key: value.default for key, value in signature.parameters.items() if value.default is not inspect.Parameter.empty}
    annotation = {key: value.annotation for key, value in signature.parameters.items() if value.annotation is not inspect.Parameter.empty}
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train', required=True, help='path to train problem set')
    parser.add_argument('--validation', help='path to validation problem set')
    parser.add_argument('--pretrained-model-name', default=default['pretrained_model_name'], help=' ')
    parser.add_argument('--log', help='log directory')
    parser.add_argument('--num-epochs', default=default['num_epochs'], type=annotation['num_epochs'], help=' ')
    parser.add_argument('--batch-size', default=default['batch_size'], type=annotation['batch_size'], help=' ')
    parser.add_argument('--lr', default=default['lr'], type=annotation['lr'], help=' ')
    parser.add_argument('--seed', default=default['seed'], type=annotation['seed'], help=' ')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--device', type=annotation['device'], help=' ')
    args = parser.parse_args()
    main(**vars(args))
