import os
import time
import math
import json
import joblib
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from functools import partial
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

from opt import adam, warmup_cosine, warmup_linear, warmup_constant
from datasets import rocstories
from analysis import rocstories as rocstories_analysis
from text_utils import TextEncoder
from utils import encode_dataset, flatten, iter_data, find_trainable_variables, get_ema_vars, convert_gradient_to_tensor, shape_list, ResultLogger, assign_to_gpu, average_grads, make_path

OPT_FNS = {
    'adam':adam,
}

LR_SCHEDULES = {
    'warmup_cosine':warmup_cosine,
    'warmup_linear':warmup_linear,
    'warmup_constant':warmup_constant,
}

class LossCompute:
    "A Loss compute and train function."
    def __init__(self, generator, lm_criterion, n_embed, opt=None):
        self.generator = generator
        self.lm_criterion = lm_criterion
        self.opt = opt
        self.n_embed = n_embed

    def __call__(self, X, Y, M, h, norm):
        # Language modeling loss
        h_trunc = h[:, :-1].contiguous().view(-1, self.n_embed) # Shape: 252, 768
        x_shifted = X[:, 1:, 0].contiguous().view(-1)           # Shape: 252
        lm_logits = self.generator(h_trunc)
        lm_losses = self.lm_criterion(h_trunc, x_shifted)
        lm_losses = lm_losses.view(x.size(0), X.size(1))
        lm_losses = lm_losses * M[:, 1:]
        lm_losses = lm_losses.sum(1) / torch.sum(M[:, 1:], 1)

        # Classification loss
        clf_h = h.view(-1, self.n_embed)

        # loss.backward()
        # if self.opt is not None:
        #     self.opt.step()
        #     self.opt.optimizer.zero_grad()
        return lm_losses

def model(X, M, Y, train=False, reuse=False):
    we = tf.get_variable("we", [n_vocab+n_special+n_ctx, n_embd], 
                         initializer=tf.random_normal_initializer(stddev=0.02))
    we = dropout(we, embd_pdrop, train)

    X = tf.reshape(X, [-1, n_ctx, 2])
    M = tf.reshape(M, [-1, n_ctx])

    h = embed(X, we)
    for layer in range(n_layer):
        h = block(h, 'h%d'%layer, train=train, scale=True)

    lm_h = tf.reshape(h[:, :-1], [-1, n_embd])
    lm_logits = tf.matmul(lm_h, we, transpose_b=True)
    lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lm_logits, labels=tf.reshape(X[:, 1:, 0], [-1]))
    lm_losses = tf.reshape(lm_losses, [shape_list(X)[0], shape_list(X)[1]-1])
    lm_losses = tf.reduce_sum(lm_losses*M[:, 1:], 1)/tf.reduce_sum(M[:, 1:], 1)

    clf_h = tf.reshape(h, [-1, n_embd])
    pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0], clf_token), tf.float32), 1), tf.int32)
    clf_h = tf.gather(clf_h, tf.range(shape_list(X)[0], dtype=tf.int32)*n_ctx+pool_idx)

    clf_h = tf.reshape(clf_h, [-1, 2, n_embd])
    if train and clf_pdrop > 0:
        shape = shape_list(clf_h)
        shape[1] = 1
        clf_h = tf.nn.dropout(clf_h, 1-clf_pdrop, shape)
    clf_h = tf.reshape(clf_h, [-1, n_embd])
    clf_logits = clf(clf_h, 1, train=train)
    clf_logits = tf.reshape(clf_logits, [-1, 2])

    clf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=clf_logits, labels=Y)
    return clf_logits, clf_losses, lm_losses

def mgpu_train(*xs):
    gpu_ops = []
    gpu_grads = []
    xs = (tf.split(x, n_gpu, 0) for x in xs)
    for i, xs in enumerate(zip(*xs)):
        do_reuse = True if i > 0 else None
        with tf.device(assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(), reuse=do_reuse):
            clf_logits, clf_losses, lm_losses = model(*xs, train=True, reuse=do_reuse)
            if lm_coef > 0:
                train_loss = tf.reduce_mean(clf_losses) + lm_coef*tf.reduce_mean(lm_losses)
            else:
                train_loss = tf.reduce_mean(clf_losses)
            params = find_trainable_variables("model")
            grads = tf.gradients(train_loss, params)
            grads = list(zip(grads, params))
            gpu_grads.append(grads)
            gpu_ops.append([clf_logits, clf_losses, lm_losses])
    ops = [tf.concat(op, 0) for op in zip(*gpu_ops)]
    grads = average_grads(gpu_grads)
    grads = [g for g, p in grads]
    train = opt_fns[opt](params, grads, lr, partial(lr_schedules[lr_schedule], warmup=lr_warmup), n_updates_total, l2=l2, max_grad_norm=max_grad_norm, vector_l2=vector_l2, b1=b1, b2=b2, e=e)
    return [train]+ops

def mgpu_predict(*xs):
    gpu_ops = []
    xs = (tf.split(x, n_gpu, 0) for x in xs)
    for i, xs in enumerate(zip(*xs)):
        with tf.device(assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(), reuse=True):
            clf_logits, clf_losses, lm_losses = model(*xs, train=False, reuse=True)
            gpu_ops.append([clf_logits, clf_losses, lm_losses])
    ops = [tf.concat(op, 0) for op in zip(*gpu_ops)]
    return ops

def transform_roc(X1, X2, X3):
    n_batch = len(X1)
    xmb = np.zeros((n_batch, 2, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, 2, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    for i, (x1, x2, x3), in enumerate(zip(X1, X2, X3)):
        x12 = [start]+x1[:max_len]+[delimiter]+x2[:max_len]+[clf_token]
        x13 = [start]+x1[:max_len]+[delimiter]+x3[:max_len]+[clf_token]
        l12 = len(x12)
        l13 = len(x13)
        xmb[i, 0, :l12, 0] = x12
        xmb[i, 1, :l13, 0] = x13
        mmb[i, 0, :l12] = 1
        mmb[i, 1, :l13] = 1
    xmb[:, :, :, 1] = np.arange(n_vocab+n_special, n_vocab+n_special+n_ctx)
    return xmb, mmb

def iter_apply(Xs, Ms, Ys):
    fns = [lambda x:np.concatenate(x, 0), lambda x:float(np.sum(x))]
    results = []
    for xmb, mmb, ymb in iter_data(Xs, Ms, Ys, n_batch=n_batch_train, truncate=False, verbose=True):
        n = len(xmb)
        if n == n_batch_train:
            res = sess.run([eval_mgpu_logits, eval_mgpu_clf_loss], {X_train:xmb, M_train:mmb, Y_train:ymb})
        else:
            res = sess.run([eval_logits, eval_clf_loss], {X:xmb, M:mmb, Y:ymb})
        res = [r*n for r in res]
        results.append(res)
    results = zip(*results)
    return [fn(res) for res, fn in zip(results, fns)]

def iter_predict(Xs, Ms):
    logits = []
    for xmb, mmb in iter_data(Xs, Ms, n_batch=n_batch_train, truncate=False, verbose=True):
        n = len(xmb)
        if n == n_batch_train:
            logits.append(sess.run(eval_mgpu_logits, {X_train:xmb, M_train:mmb}))
        else:
            logits.append(sess.run(eval_logits, {X:xmb, M:mmb}))
    logits = np.concatenate(logits, 0)
    return logits

def save(path):
    ps = sess.run(params)
    joblib.dump(ps, make_path(path))

def log():
    global best_score
    tr_logits, tr_cost = iter_apply(trX[:n_valid], trM[:n_valid], trY[:n_valid])
    va_logits, va_cost = iter_apply(vaX, vaM, vaY)
    tr_cost = tr_cost/len(trY[:n_valid])
    va_cost = va_cost/n_valid
    tr_acc = accuracy_score(trY[:n_valid], np.argmax(tr_logits, 1))*100.
    va_acc = accuracy_score(vaY, np.argmax(va_logits, 1))*100.
    logger.log(n_epochs=n_epochs, n_updates=n_updates, tr_cost=tr_cost, va_cost=va_cost, tr_acc=tr_acc, va_acc=va_acc)
    print('%d %d %.3f %.3f %.2f %.2f'%(n_epochs, n_updates, tr_cost, va_cost, tr_acc, va_acc))
    if submit:
        score = va_acc
        if score > best_score:
            best_score = score
            save(os.path.join(save_dir, desc, 'best_params.jl'))

argmax = lambda x:np.argmax(x, 1)

pred_fns = {
    'rocstories':argmax,
}

filenames = {
    'rocstories':'ROCStories.tsv',
}

label_decoders = {
    'rocstories':None,
}

def predict():
    filename = filenames[dataset]
    pred_fn = pred_fns[dataset]
    label_decoder = label_decoders[dataset]
    predictions = pred_fn(iter_predict(teX, teM))
    if label_decoder is not None:
        predictions = [label_decoder[prediction] for prediction in predictions]
    path = os.path.join(submission_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('{}\t{}\n'.format('index', 'prediction'))
        for i, prediction in enumerate(predictions):
            f.write('{}\t{}\n'.format(i, prediction))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--save_dir', type=str, default='save/')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--submission_dir', type=str, default='submission/')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=3)
    parser.add_argument('--n_batch', type=int, default=8)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--n_ctx', type=int, default=512)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--clf_pdrop', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--n_gpu', type=int, default=4)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--encoder_path', type=str, default='model/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default='model/vocab_40000.bpe')
    parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)

    args = parser.parse_args()
    print(args)
    globals().update(args.__dict__)
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)), **args.__dict__)
    text_encoder = TextEncoder(encoder_path, bpe_path)
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)

    (trX1, trX2, trX3, trY), (vaX1, vaX2, vaX3, vaY), (teX1, teX2, teX3) = encode_dataset(rocstories(data_dir), encoder=text_encoder)
    n_y = 2
    encoder['_start_'] = len(encoder)
    encoder['_delimiter_'] = len(encoder)
    encoder['_classify_'] = len(encoder)
    clf_token = encoder['_classify_']
    n_special = 3
    max_len = n_ctx//2-2
    n_ctx = min(
                max(
                    [len(x1[:max_len])+max(len(x2[:max_len]), len(x3[:max_len])) for x1, x2, x3 in zip(trX1, trX2, trX3)]
                    +[len(x1[:max_len])+max(len(x2[:max_len]), len(x3[:max_len])) for x1, x2, x3 in zip(vaX1, vaX2, vaX3)]
                    +[len(x1[:max_len])+max(len(x2[:max_len]), len(x3[:max_len])) for x1, x2, x3 in zip(teX1, teX2, teX3)]
                   )+3, n_ctx
                )
    trX, trM = transform_roc(trX1, trX2, trX3)
    vaX, vaM = transform_roc(vaX1, vaX2, vaX3)
    if submit:
        teX, teM = transform_roc(teX1, teX2, teX3)

    n_train = len(trY)
    n_valid = len(vaY)
    n_batch_train = n_batch*n_gpu
    n_updates_total = (n_train//n_batch_train)*n_iter

    X_train = tf.placeholder(tf.int32, [n_batch_train, 2, n_ctx, 2])
    M_train = tf.placeholder(tf.float32, [n_batch_train, 2, n_ctx])
    X = tf.placeholder(tf.int32, [None, 2, n_ctx, 2])
    M = tf.placeholder(tf.float32, [None, 2, n_ctx])

    Y_train = tf.placeholder(tf.int32, [n_batch_train])
    Y = tf.placeholder(tf.int32, [None])

    train, logits, clf_losses, lm_losses = mgpu_train(X_train, M_train, Y_train)
    clf_loss = tf.reduce_mean(clf_losses)

    params = find_trainable_variables('model')
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())

    shapes = json.load(open('model/params_shapes.json'))
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    init_params = [np.load('model/params_{}.npy'.format(n)) for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
    init_params[0] = init_params[0][:n_ctx]
    init_params[0] = np.concatenate([init_params[1], (np.random.randn(n_special, n_embd)*0.02).astype(np.float32), init_params[0]], 0)
    del init_params[1]

    if n_transfer == -1:
        n_transfer = 0
    else:
        n_transfer = 1+n_transfer*12
    sess.run([p.assign(ip) for p, ip in zip(params[:n_transfer], init_params[:n_transfer])])

    eval_mgpu_logits, eval_mgpu_clf_losses, eval_mgpu_lm_losses = mgpu_predict(X_train, M_train, Y_train)
    eval_logits, eval_clf_losses, eval_lm_losses = model(X, M, Y, train=False, reuse=True)
    eval_clf_loss = tf.reduce_mean(eval_clf_losses)
    eval_mgpu_clf_loss = tf.reduce_mean(eval_mgpu_clf_losses)

    n_updates = 0
    n_epochs = 0
    if dataset != 'stsb':
        trYt = trY
    if submit:
        save(os.path.join(save_dir, desc, 'best_params.jl'))
    best_score = 0
    for i in range(n_iter):
        for xmb, mmb, ymb in iter_data(*shuffle(trX, trM, trYt, random_state=np.random), n_batch=n_batch_train, truncate=True, verbose=True):
            cost, _ = sess.run([clf_loss, train], {X_train:xmb, M_train:mmb, Y_train:ymb})
            n_updates += 1
            if n_updates in [1000, 2000, 4000, 8000, 16000, 32000] and n_epochs == 0:
                log()
        n_epochs += 1
        log()
    if submit:
        sess.run([p.assign(ip) for p, ip in zip(params, joblib.load(os.path.join(save_dir, desc, 'best_params.jl')))])
        predict()
        if analysis:
            rocstories_analysis(data_dir, os.path.join(submission_dir, 'ROCStories.tsv'), os.path.join(log_dir, 'rocstories.jsonl'))
