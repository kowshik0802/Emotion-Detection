import tensorflow as tf2

from tf2 import model

def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf2.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf2.newaxis]
        return tf2.compat.v1.where(
            logits < min_values,
            tf2.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf2.cond(
       pred=tf2.equal(k, 0),
       true_fn=lambda: logits,
       false_fn=lambda: _top_k(),
    )


def top_p_logits(logits, p):
    """Nucleus sampling"""
    batch, _ = logits.shape.as_list()
    sorted_logits = tf2.sort(logits, direction='DESCENDING', axis=-1)
    cumulative_probs = tf2.cumsum(tf2.nn.softmax(sorted_logits, axis=-1), axis=-1)
    indices = tf2.stack([
        tf2.range(0, batch),
        # number of indices to include
        tf2.maximum(tf2.reduce_sum(input_tensor=tf2.cast(cumulative_probs <= p, tf2.int32), axis=-1) - 1, 0),
    ], axis=-1)
    min_values = tf2.gather_nd(sorted_logits, indices)
    return tf2.compat.v1.where(
        logits < min_values,
        tf2.ones_like(logits) * -1e10,
        logits,
    )


def sample_sequence(*, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, top_p=1):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf2.fill([batch_size, 1], start_token)

    def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf2.compat.v1.AUTO_REUSE)

        logits = lm_output['logits'][:, :, :hparams['n_vocab']]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf2.compat.v1.name_scope('sample_sequence'):
        def body(past, prev, output):
            next_outputs = step(hparams, prev, past=past)
            logits = next_outputs['logits'][:, -1, :]  / tf2.cast(temperature, dtype=tf2.float32)
            logits = top_k_logits(logits, k=top_k)
            logits = top_p_logits(logits, p=top_p)
            samples = tf2.random.categorical(logits=logits, num_samples=1, dtype=tf2.int32)
            return [
                next_outputs['presents'] if past is None else tf2.concat([past, next_outputs['presents']], axis=-2),
                samples,
                tf2.concat([output, samples], axis=1)
            ]

        past, prev, output = body(None, context, context)

        def cond(*args):
            return True

        _, _, tokens = tf2.while_loop(
            cond=cond, body=body,
            maximum_iterations=length - 1,
            loop_vars=[
                past,
                prev,
                output
            ],
            shape_invariants=[
                tf2.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
                tf2.TensorShape([batch_size, None]),
                tf2.TensorShape([batch_size, None]),
            ],
            back_prop=False,
        )

        return tf2.stop_gradient(tokens)
