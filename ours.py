
import torch.nn.functional as F
from transformers import (AutoTokenizer, DistilBertForSequenceClassification,
                          AutoModelForSequenceClassification, get_scheduler,
                          DataCollatorWithPadding)
from torch.optim import AdamW
import torch.nn as nn
from torch import masked_select

#--tinybert framework
import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from transformer.modeling import TinyBertForSequenceClassification
from transformer.tokenization import BertTokenizer
from transformer.optimization import BertAdam
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME


#--load class
from codes.custom import CustomBertSmall
from codes.cka import CKALoss, CKASimilarity
from codes.processors import (
        ColaProcessor,
        MnliProcessor,
        MnliMismatchedProcessor,
        MrpcProcessor,
        Sst2Processor,
        StsbProcessor,
        QqpProcessor,
        QnliProcessor,
        RteProcessor,
        WnliProcessor
)


csv.field_size_limit(sys.maxsize)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

fh = logging.FileHandler('debug_layer_loss.log')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()

#--------------------------------------------------------
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, seq_length=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.seq_length = seq_length
        self.label_id = label_id


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        seq_length = len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(example.label))
            logger.info("label_id: {}".format(label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          seq_length=seq_length))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def get_tensor_data(output_mode, features):
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_label_ids, all_seq_lengths)
    return tensor_data, all_label_ids


def result_to_file(result, file_name):
    with open(file_name, "a") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))


def do_eval(model, task_name, eval_dataloader,
            device, output_mode, eval_labels, num_labels):
    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for batch_ in tqdm(eval_dataloader, desc="Evaluating"):
        batch_ = tuple(t.to(device) for t in batch_)
        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_

            logits, _ = model(input_ids, segment_ids, input_mask, output_hidden_states=True)
            print("🔍 logits sample:", logits)
        # create eval loss and other metric required by the task
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    preds = preds[0]
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(task_name, preds, eval_labels.numpy())
    result['eval_loss'] = eval_loss
    print("🔍 preds:", preds)


    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default="/mnt/aix7101/jeong/glue_data",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--teacher_model",
                        default=None,
                        type=str,
                        help="The teacher model dir.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default="/mnt/aix7101/jeong/ee/Checkpoints",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay', '--wd',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay')
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    # added arguments
    parser.add_argument('--aug_train',
                        action='store_true')
    parser.add_argument('--eval_step',
                        type=int,
                        default=50)
    parser.add_argument('--pred_distill',
                        action='store_true')
    parser.add_argument('--data_url',
                        type=str,
                        default="")
    parser.add_argument('--temperature',
                        type=float,
                        default=1.)

    args = parser.parse_args()
    logger.info('The args: {}'.format(args))

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mnli-mm": MnliMismatchedProcessor,
        "mrpc": MrpcProcessor,
        "sst-2": Sst2Processor,
        "sts-b": StsbProcessor,
        "qqp": QqpProcessor,
        "qnli": QnliProcessor,
        "rte": RteProcessor,
        "wnli": WnliProcessor
    }

    output_modes = {
        "cola": "classification",
        "mnli": "classification",
        "mrpc": "classification",
        "sst-2": "classification",
        "sts-b": "regression",
        "qqp": "classification",
        "qnli": "classification",
        "rte": "classification",
        "wnli": "classification"
    }

    # intermediate distillation default parameters
    default_params = {
        "cola": {"num_train_epochs": 50, "max_seq_length": 64},
        "mnli": {"num_train_epochs": 5, "max_seq_length": 128},
        "mrpc": {"num_train_epochs": 20, "max_seq_length": 128},
        "sst-2": {"num_train_epochs": 10, "max_seq_length": 64},
        "sts-b": {"num_train_epochs": 20, "max_seq_length": 128},
        "qqp": {"num_train_epochs": 5, "max_seq_length": 128},
        "qnli": {"num_train_epochs": 10, "max_seq_length": 128},
        "rte": {"num_train_epochs": 20, "max_seq_length": 128}
    }

    acc_tasks = ["mnli", "mrpc", "sst-2", "qqp", "qnli", "rte"]
    corr_tasks = ["sts-b"]
    mcc_tasks = ["cola"]

    # Prepare devices
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    # Prepare seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Prepare task settings
    # if os.path.exists(args.output_dir + "/" + args.task_name) and os.listdir(args.output_dir + "/" + args.task_name):
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir + "/" + args.task_name))
    if not os.path.exists(args.output_dir + "/" + args.task_name):
        os.makedirs(args.output_dir + "/" + args.task_name)

    task_name = args.task_name.lower()

    if task_name in default_params:
        args.max_seq_len = default_params[task_name]["max_seq_length"]

    if not args.pred_distill and not args.do_eval:
        if task_name in default_params:
            args.num_train_epoch = default_params[task_name]["num_train_epochs"]

    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)

    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.teacher_model, do_lower_case=args.do_lower_case)

    if not args.do_eval:
        if not args.aug_train:
            train_examples = processor.get_train_examples(args.data_dir + "/" + args.task_name)
        else:
            train_examples = processor.get_aug_examples(args.data_dir + "/" + args.task_name)
        if args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps))

        args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

        train_features = convert_examples_to_features(train_examples, label_list,
                                                      args.max_seq_length, tokenizer, output_mode)
        train_data, _ = get_tensor_data(output_mode, train_features)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_examples = processor.get_dev_examples(args.data_dir + "/" + args.task_name)
    eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
    eval_data, eval_labels = get_tensor_data(output_mode, eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    #---------------
    cka_weight = 1.0
    #---------------

    teacher_model = AutoModelForSequenceClassification.from_pretrained(args.teacher_model, num_labels=num_labels)
    teacher_model.to(device)
    
    total_l, trans_l = 6, 1
    student_model = CustomBertSmall(teacher_model=teacher_model, total_layers=total_l, transplanted_layers=trans_l)
    student_model.to(device)

    if args.do_eval:
        # load model
        model_path = os.path.join(args.output_dir, args.task_name, "pytorch_model.bin")
        student_model.load_state_dict(torch.load(model_path, map_location=device))
        student_model.to(device)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        student_model.eval()
        result = do_eval(student_model, task_name, eval_dataloader,
                         device, output_mode, eval_labels, num_labels)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    else:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        if n_gpu > 1:
            student_model = torch.nn.DataParallel(student_model)
            teacher_model = torch.nn.DataParallel(teacher_model)
        # Prepare optimizer
        param_optimizer = list(student_model.named_parameters())
        size = 0
        for n, p in student_model.named_parameters():
            logger.info('n: {}'.format(n))
            size += p.nelement()

        logger.info('Total parameters: {}'.format(size))
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        schedule = 'warmup_linear'
        if not args.pred_distill:
            schedule = 'none'
        optimizer = BertAdam(optimizer_grouped_parameters,
                             schedule=schedule,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
        # Prepare loss functions
        loss_mse = MSELoss()

        def soft_cross_entropy(predicts, targets):
            student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
            targets_prob = torch.nn.functional.softmax(targets, dim=-1)
            return (- targets_prob * student_likelihood).mean()

        # Train and evaluate
        global_step = 0
        best_dev_acc = 0.0
        output_eval_file = os.path.join(args.output_dir + "/" + args.task_name, "eval_results.txt")

        ckaloss = CKALoss()
        stloss = nn.KLDivLoss(reduction='batchmean')

        for epoch_ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0.

            tr_cka_loss = 0.
            tr_cls_loss = 0.
            tr_kd_loss = 0.

            student_model = student_model.to(device)
            student_model.train()

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
                # # for check parameter updating
                # if step == 0 and epoch_ == 0:
                #     tracked_param_name = 'module.encoder_layers.0.attention.output.dense.weight'
                #     tracked_param_before = student_model.state_dict()[tracked_param_name].clone().detach()                
                # #--------------------------
                                
                batch = tuple(t.to(device) for t in batch)

                input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
                if input_ids.size()[0] != args.train_batch_size:
                    continue

                cka_loss = 0.
                cls_loss = 0.
                kd_loss = 0.
                
                student_logits, student_reps = student_model(input_ids, segment_ids, input_mask, output_hidden_states=True)

                with torch.no_grad():
                    teacher_outputs = teacher_model(input_ids, segment_ids, input_mask, output_hidden_states=True)
                teacher_logits, teacher_reps = teacher_outputs.logits, teacher_outputs.hidden_states
                
                input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch

                student_mask = input_mask.unsqueeze(-1).expand_as(student_reps[-1]).bool()
                teacher_mask = input_mask.unsqueeze(-1).expand_as(teacher_reps[-1]).bool()

                student_layers = [0, 1, 2, 3, 4, 5]
                teacher_layers = [1, 3, 5, 7, 9, 11]

                for s_idx, t_idx in zip(student_layers, teacher_layers):
                    SH = masked_select(student_reps[s_idx], student_mask).view(-1, 768)
                    TH = masked_select(teacher_reps[t_idx+1], teacher_mask).view(-1, 768)
                    cka_loss += ckaloss(SH, TH)
                tr_cka_loss += cka_loss.item()

                if output_mode == "classification":
                    cls_loss = soft_cross_entropy(student_logits / args.temperature,
                                                    teacher_logits / args.temperature)
                elif output_mode == "regression":
                    loss_mse = MSELoss()
                    cls_loss = loss_mse(student_logits.view(-1), label_ids.view(-1))
                tr_cls_loss += cls_loss.item()

                kd_loss = stloss(F.log_softmax(student_logits, dim=-1), F.softmax(teacher_logits, dim=-1))
                tr_kd_loss += kd_loss.item()

                loss = cka_loss + cls_loss + kd_loss

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if (global_step + 1) % args.eval_step == 0:
                    logger.info("***** Running evaluation *****")
                    logger.info("  Process = {} iter {} step".format(epoch_, global_step))
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)

                    student_model.eval()

                    loss = tr_loss / (step + 1)

                    cka_loss = tr_cka_loss / (step + 1)
                    cls_loss = tr_cls_loss / (step + 1)
                    kd_loss = tr_kd_loss / (step + 1)

                    result = do_eval(student_model, task_name, eval_dataloader,
                                     device, output_mode, eval_labels, num_labels)

                    result['global_step'] = global_step
                    result['loss'] = loss
                    result['cka_loss'] = cka_loss
                    result['cls_loss'] = cls_loss
                    result['kd_loss'] = kd_loss

                    result_to_file(result, output_eval_file)

                    if not args.pred_distill:
                        save_model = True
                    else:
                        save_model = False

                        if task_name in acc_tasks and result['acc'] > best_dev_acc:
                            best_dev_acc = result['acc']
                            save_model = True

                        if task_name in corr_tasks and result['corr'] > best_dev_acc:
                            best_dev_acc = result['corr']
                            save_model = True

                        if task_name in mcc_tasks and result['mcc'] > best_dev_acc:
                            best_dev_acc = result['mcc']
                            save_model = True

                    # # for check parameter updating

                    # tracked_param_after = student_model.state_dict()[tracked_param_name].clone().detach()
                    # param_diff = (tracked_param_after - tracked_param_before).abs().mean().item()
                    # print(f"🔍 Param '{tracked_param_name}' mean diff after step {step+1}: {param_diff:.6f}")
                    # tracked_param_before = tracked_param_after  # 갱신
                    # #------------------

                    if save_model:
                        logger.info("***** Save model *****")

                        model_to_save = student_model.module if hasattr(student_model, 'module') else student_model

                        model_name = WEIGHTS_NAME
                        # if not args.pred_distill:
                        #     model_name = "step_{}_{}".format(global_step, WEIGHTS_NAME)
                        output_model_file = os.path.join(args.output_dir + "/" + args.task_name, model_name)
                        output_config_file = os.path.join(args.output_dir + "/" + args.task_name, CONFIG_NAME)

                        torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        tokenizer.save_vocabulary(args.output_dir + "/" + args.task_name)

                        # Test mnli-mm
                        if args.pred_distill and task_name == "mnli":
                            task_name = "mnli-mm"
                            processor = processors[task_name]()
                            if not os.path.exists(args.output_dir + "/" + args.task_name + '-MM'):
                                os.makedirs(args.output_dir + "/" + args.task_name + '-MM')

                            eval_examples = processor.get_dev_examples(args.data_dir + "/" + args.task_name)

                            eval_features = convert_examples_to_features(
                                eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
                            eval_data, eval_labels = get_tensor_data(output_mode, eval_features)

                            logger.info("***** Running mm evaluation *****")
                            logger.info("  Num examples = %d", len(eval_examples))
                            logger.info("  Batch size = %d", args.eval_batch_size)

                            eval_sampler = SequentialSampler(eval_data)
                            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                                                         batch_size=args.eval_batch_size)

                            result = do_eval(student_model, task_name, eval_dataloader,
                                             device, output_mode, eval_labels, num_labels)

                            result['global_step'] = global_step

                            tmp_output_eval_file = os.path.join(args.output_dir + "/" + args.task_name + '-MM', "eval_results.txt")
                            result_to_file(result, tmp_output_eval_file)

                            task_name = 'mnli'

                    student_model.train()


if __name__ == "__main__":
    main()
