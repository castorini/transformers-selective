import os
import time
import logging
import random
from tqdm import tqdm, trange
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import (TensorDataset, DataLoader,
                              RandomSampler, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import glue_compute_metrics as compute_metrics
from transformers.selective_metrics import rcc_auc, rpp
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors

from torchprofile.profile import profile_macs

logger = logging.getLogger(__name__)


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    if task in ['mnli-mm', 'bmnli-mm']:
        data_dir = args.data_dir[:-3]
    else:
        data_dir = args.data_dir
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(data_dir) if evaluate else processor.get_train_examples(data_dir)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    if (args.train_percentage<100) and (not evaluate):
        features = features[:int(len(features) * args.train_percentage / 100)]
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode in ["classification", "selective-classification"]:
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    all_indices = torch.tensor(range(len(features)), dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_indices)
    return dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_wanted_result(result):
    if "spearmanr" in result:
        print_result = result["spearmanr"]
    elif "f1" in result:
        print_result = result["f1"]
    elif "mcc" in result:
        print_result = result["mcc"]
    elif "acc" in result:
        print_result = result["acc"]
    else:
        print(result)
        exit(1)
    return print_result


def get_dataset_acc(args, dataset, model):
    # routine doesn't matter since it only affects the loss
    eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    logits = []
    out_label_ids = []
    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert',
                                                                           'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, batch_logits = outputs[:3]
        logits.append(batch_logits.detach().cpu().numpy())
        out_label_ids.append(inputs['labels'].detach().cpu().numpy())

    logits = np.concatenate(logits, axis=0)
    out_label_ids = np.concatenate(out_label_ids, axis=0)
    if args.output_mode in ["classification", "selective-classification"]:
        preds = np.argmax(logits, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(logits)

    return preds == out_label_ids  # only works for classification


def train(args, train_dataset, model, tokenizer, routine='raw'):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    if routine.endswith('hist'):
        correctness_record = [[0, 0] for _ in range(len(train_dataset))]
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for i_epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            if (
                    routine.endswith('hist')
                    and step % (len(epoch_iterator) // 10) == 0
                    and step / len(epoch_iterator) <= 0.9
            ):
                # update correctness record
                new_record = get_dataset_acc(args, train_dataset, model)
                for i in range(len(correctness_record)):
                    correctness_record[i][0] += 1  # denominator
                    correctness_record[i][1] += new_record[i]  # numerator

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[3],
                'routine': routine,
                'i_epoch': i_epoch,
                'step': step,
            }
            if routine.endswith('hist'):
                inputs['history_record'] = torch.tensor([
                    correctness_record[ind][1] / correctness_record[ind][0] for ind in batch[-1]
                ], device=args.device)
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def mc_dropout(args, eval_dataloader, model, runs=10):
    # the model originally has a dropout rate of 0.1, but set to 0.01 for mc_dropout
    logits = []
    for mc_index in range(runs):
        run_logits = []
        torch.manual_seed(42 * (1+mc_index))  # for different dropout seeds
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.train()  # use train() delibrately so dropout is applied
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'labels': batch[3],
                }
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert',
                                                                               'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                _, batch_logits = outputs[:3]
            run_logits.append(batch_logits.detach().cpu().numpy())
        logits.append(np.concatenate(run_logits, axis=0))
    raw_logits = np.array(logits)
    max_logits = np.max(raw_logits, axis=2).transpose(1, 0)
    return raw_logits, max_logits


def evaluate(args, model, tokenizer, prefix="", routine='raw'):
    # Loop to handle MNLI double evaluation (matched, mis-matched)

    eval_task = args.task_name
    eval_output_dir = args.output_dir + ("-MM" if "mm" in args.task_name else "")

    results = {}
    eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    logits = []
    probs = []
    out_label_ids = []
    start_time = time.time()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[3],
                'routine': routine,
            }
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, batch_logits = outputs[:3]
            batch_probs = F.softmax(batch_logits, dim=1)
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        logits.append(batch_logits.detach().cpu().numpy())
        probs.append(batch_probs.detach().cpu().numpy())
        out_label_ids.append(inputs['labels'].detach().cpu().numpy())

    eval_time = time.time() - start_time
    eval_loss = eval_loss / nb_eval_steps
    logits = np.concatenate(logits, axis=0)
    probs = np.concatenate(probs, axis=0)
    out_label_ids = np.concatenate(out_label_ids, axis=0)

    if args.output_mode == "selective-classification":
        th_results = {}  # threshold -> score
        for threshold in np.arange(0.7, 1.0, 0.001):
            preds = np.argmax(logits, axis=1)
            for i in range(len(preds)):
                if max(probs[i]) < threshold:
                    preds[i] = model.num_labels - 1
            th_results[threshold] = compute_metrics(eval_task, preds, out_label_ids)['acc']

        best_th, best_acc = 0, 0
        for th, acc in th_results.items():
            # print(th, acc)
            if acc > best_acc:
                best_th, best_acc = th, acc
        print({
            'best_th': best_th,
            'best_acc': best_acc,
        })

    if args.output_mode in ["classification", "selective-classification"]:
        preds = np.argmax(logits, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(logits)
    result = compute_metrics(eval_task, preds, out_label_ids)
    print_result = get_wanted_result(result)

    conf = probs.max(axis=1)
    risk_binary = (preds != out_label_ids).astype(int)
    rcc_auc_value = rcc_auc(conf, risk_binary, 'SR', args)
    rpp_value = rpp(conf, risk_binary)

    print({
        "eval_time": eval_time,
        "result": print_result,
        "rcc_auc": rcc_auc_value,
        "rpp": rpp_value,
    })
    results.update(result)

    if args.top2_diff:
        sorted_probs = np.sort(probs, axis=1)
        t2d_conf = sorted_probs[:, -1] - sorted_probs[:, -2]
        t2d_rcc_auc_value = rcc_auc(t2d_conf, risk_binary, None, args)
        t2d_rpp_value = rpp(t2d_conf, risk_binary)
        print({
            't2d_rcc_auc': t2d_rcc_auc_value,
            't2d_rpp': t2d_rpp_value,
        })

    if args.mc_dropout:
        _, mc_max_logits = mc_dropout(args, eval_dataloader, model)
        mc_conf = -np.var(mc_max_logits, axis=1)
        mc_rcc_auc_value = rcc_auc(mc_conf, risk_binary, 'MC', args)
        mc_rpp_value = rpp(mc_conf, risk_binary)
        print({
            'mc_rcc_auc': mc_rcc_auc_value,
            'mc_rpp': mc_rpp_value,
        })

    if args.multi_mc_dropout:
        save_fname = build_file(
            args.plot_data_dir,
            'saved_data',
            os.path.basename(args.data_dir),
            args.model_name_or_path + '-mc_dropout.npy'
        )
        record_into_file(save_fname, 'sr', rcc_auc_value)

        result_dict = {}
        runs = [2, 5, 10, 20, 50, 100]
        for run in runs:
            _, mc_max_logits = mc_dropout(args, eval_dataloader, model, run)
            mc_conf = -np.var(mc_max_logits, axis=1)
            mc_rcc_auc_value = rcc_auc(mc_conf, risk_binary)
            # mc_rpp_value = rpp(mc_conf, risk_binary)
            result_dict[run] = mc_rcc_auc_value
            print(run, mc_rcc_auc_value)
        record_into_file(save_fname, args.dropout_prob, result_dict)

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def multi_stage_evaluate(args, model, super_model, tokenizer,
                         prefix="", profile=True):
    # Loop to handle MNLI double evaluation (matched, mis-matched)

    eval_task = args.task_name
    eval_output_dir = args.output_dir + ("-MM" if "mm" in args.task_name else "")

    results = {}
    eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        super_model = torch.nn.DataParallel(super_model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    nb_eval_steps = 0
    logits = []
    probs = []
    out_label_ids = []
    all_macs = []
    total_goto_super = 0
    start_time = time.time()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        super_model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            # go through the base model (lstm) first
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[3],
            }
            if args.model_type in ['bert', 'xlnet']:
                inputs['token_type_ids'] = batch[2]  # XLM, DistilBERT and RoBERTa don't use segment_ids
            if profile:
                macs, batch_logits = profile_macs(model, batch[:3])
                batch_logits = batch_logits[0]
                all_macs.append(macs)
            else:
                _, batch_logits = model(**inputs)
                batch_logits = batch_logits[0]
            batch_probs = F.softmax(batch_logits, dim=1)

            # decide which samples to send to super_model
            if args.random_multi_stage_proportion is None:
                goto_super = batch_probs.max(dim=1)[0] < args.conf_th  # whether to sent a sample to super_model
            else:
                # just the first few samples in the batch
                N = len(batch[0])
                goto_super = torch.tensor(
                    [(i/N < args.random_multi_stage_proportion) for i in range(N)]
                ).to(batch[0].device)
            goto_super_count = sum(goto_super)
            total_goto_super += goto_super_count

            # go through the super model (bert)
            if goto_super_count > 0:
                picked_inputs = {
                    'input_ids': batch[0][goto_super],
                    'attention_mask': batch[1][goto_super],
                    'labels': batch[3][goto_super],
                }
                if args.model_type in ['bert', 'xlnet']:
                    picked_inputs['token_type_ids'] = batch[2][goto_super]  # XLM, DistilBERT and RoBERTa don't use segment_ids
                if profile:
                    macs, super_batch_logits = profile_macs(
                        super_model,
                        (batch[0][goto_super], batch[1][goto_super], batch[2][goto_super])
                    )
                    super_batch_logits = super_batch_logits[0]
                    all_macs.append(macs)
                else:
                    _, super_batch_logits = super_model(**picked_inputs)
                super_batch_probs = F.softmax(super_batch_logits, dim=1)

        nb_eval_steps += 1

        logits.append(batch_logits[~goto_super].detach().cpu().numpy())
        probs.append(batch_probs[~goto_super].detach().cpu().numpy())
        out_label_ids.append(batch[3][~goto_super].detach().cpu().numpy())
        if goto_super_count > 0:
            logits.append(super_batch_logits.detach().cpu().numpy())
            probs.append(super_batch_probs.detach().cpu().numpy())
            out_label_ids.append(batch[3][goto_super].detach().cpu().numpy())

    eval_time = time.time() - start_time
    logits = np.concatenate(logits, axis=0)
    probs = np.concatenate(probs, axis=0)
    out_label_ids = np.concatenate(out_label_ids, axis=0)

    if args.output_mode == "selective-classification":
        th_results = {}  # threshold -> score
        for threshold in np.arange(0.7, 1.0, 0.001):
            preds = np.argmax(logits, axis=1)
            for i in range(len(preds)):
                if max(probs[i]) < threshold:
                    preds[i] = model.num_labels - 1
            th_results[threshold] = compute_metrics(eval_task, preds, out_label_ids)['acc']

        best_th, best_acc = 0, 0
        for th, acc in th_results.items():
            # print(th, acc)
            if acc > best_acc:
                best_th, best_acc = th, acc
        print({
            'best_th': best_th,
            'best_acc': best_acc,
        })

    if args.output_mode in ["classification", "selective-classification"]:
        preds = np.argmax(logits, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(logits)
    result = compute_metrics(eval_task, preds, out_label_ids)
    print_result = get_wanted_result(result)

    conf = probs.max(axis=1)
    risk_binary = (preds != out_label_ids).astype(int)
    rcc_auc_value = rcc_auc(conf, risk_binary)
    rpp_value = rpp(conf, risk_binary)

    result_dict = {
        "goto_super": float(total_goto_super) / len(eval_dataset),
        "eval_time": eval_time,
        "result": print_result,
        "rcc_auc": rcc_auc_value,
        "rpp": rpp_value,
        "macs(G)": sum(all_macs) / len(out_label_ids) / 1e9,
    }

    print(result_dict)
    results.update(result)

    save_fname = build_file(
        args.plot_data_dir,
        'saved_data',
        os.path.basename(args.data_dir),
        ('' if args.random_multi_stage_proportion is None else 'random-') + \
        args.multi_stage_base + '-' + args.train_routine + '.npy'
    )
    record_into_file(save_fname, args.conf_th, result_dict)

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def build_file(*args):
    save_fname = os.path.join(*args)
    if not os.path.exists(save_fname):
        os.makedirs(os.path.dirname(save_fname), exist_ok=True)
    return save_fname


def record_into_file(fname, key, value):
    if os.path.exists(fname):
        saved_dict = np.load(fname, allow_pickle=True).item()
    else:
        saved_dict = {}

    if key in saved_dict:
        print(key, 'already in the file!')
    else:
        saved_dict[key] = value
        np.save(fname, saved_dict)
